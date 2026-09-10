import unittest

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from domain_mf.adaptive import AdaptiveCandidate, select_adaptive_initialisation
from domain_mf.faithfulness import (
    decompose_logits,
    intervened_logits,
    mask_jaccard,
    target_contributions,
    topk_mask,
)
from domain_mf.initializers import calibrate_logit_scale, initialise_model
from domain_mf.interpretability import (
    anchor_regularisation,
    capture_anchors,
    kernel_drift,
    template_assignment_stability,
)
from domain_mf.models import (
    C4RotationInvariantCNN,
    CorrEncoderCNN,
    FullCorrEncoderCNN,
    HalfWidthCNN,
    MFOneLayer,
    MFTwoLayer,
    ModelSpec,
    QuarterWidthCNN,
    PublishedCorrEncoder1D,
    SharedConceptCNN,
    SmallResNet,
    StandardCNN,
    StandardResNet18,
)
from domain_mf.regression import (
    SafeAffineCalibration,
    calibrate_regression_output,
    correncoder_regression_loss,
    fit_safe_calibration_head,
    initialise_1d_matched_filter,
    initialise_1d_layerwise_matched_filter,
    lag_tolerant_pearson_loss,
    pearson_correlation_loss,
    soft_lag_pearson_loss,
    spectral_shape_loss,
)
from evaluate_pruning import apply_global_pruning
from evaluate_structured_efficiency import physically_prune
from evaluate_reliability import binary_auc, classification_metrics
from run_correncoder_regression import lag_robust_waveform_correlation


def synthetic_loader():
    images = torch.zeros(20, 1, 4, 4)
    labels = torch.tensor([0, 1] * 10)
    images[labels == 0, :, :2, :] = 1.0
    images[labels == 1, :, 2:, :] = 1.0
    images += 0.02 * torch.randn_like(images)
    return DataLoader(TensorDataset(images, labels), batch_size=5, shuffle=False)


def large_synthetic_loader():
    images = torch.zeros(100, 1, 4, 4)
    labels = torch.tensor([0, 1] * 50)
    images[labels == 0, :, :2, :] = 1.0
    images[labels == 1, :, 2:, :] = 1.0
    images += 0.02 * torch.randn_like(images)
    return DataLoader(TensorDataset(images, labels), batch_size=20, shuffle=False)


class InitialisationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.spec = ModelSpec("synthetic", 4, 1, 2, 2, 1)
        self.loader = synthetic_loader()
        self.device = torch.device("cpu")

    def test_one_layer_di_wmf_is_finite_and_discriminative(self):
        model = MFOneLayer(self.spec)
        report = initialise_model(model, self.loader, "di_wmf", self.device)
        self.assertEqual(report.layers_initialised, 1)
        self.assertTrue(torch.isfinite(model.conv1.weight).all())
        images, labels = next(iter(self.loader))
        accuracy = (model(images).argmax(1) == labels).float().mean()
        self.assertGreaterEqual(float(accuracy), 0.8)

    def test_two_layer_shapes_and_anchor_metrics(self):
        model = MFTwoLayer(self.spec)
        report = initialise_model(model, self.loader, "di_wmf", self.device, layers=2)
        self.assertEqual(report.layers_initialised, 2)
        self.assertEqual(tuple(model.conv1.weight.shape), (18, 1, 2, 2))
        self.assertEqual(tuple(model.conv2.weight.shape), (2, 18, 3, 3))
        anchors = capture_anchors(model)
        self.assertAlmostEqual(
            float(anchor_regularisation(model, anchors).detach()), 0.0, places=6
        )
        self.assertAlmostEqual(kernel_drift(model, anchors)["conv1"]["mean"], 0.0, places=6)
        self.assertEqual(template_assignment_stability(model, anchors)["conv1"], 1.0)

    def test_classmean_initialises_both_layers(self):
        model = MFTwoLayer(self.spec)
        initialise_model(model, self.loader, "classmean", self.device, layers=2)
        self.assertTrue(torch.isfinite(model(torch.randn(3, 1, 4, 4))).all())

    def test_logit_calibration_preserves_predictions(self):
        model = MFTwoLayer(self.spec)
        initialise_model(model, self.loader, "di_wmf", self.device, layers=2)
        images, _ = next(iter(self.loader))
        before = model(images).detach()
        scale = calibrate_logit_scale(
            model, self.loader, self.device, target_std=1.0
        )
        after = model(images).detach()
        self.assertGreater(scale, 0.0)
        self.assertTrue(torch.equal(before.argmax(1), after.argmax(1)))
        all_logits = torch.cat([model(x) for x, _ in self.loader])
        self.assertAlmostEqual(float(all_logits.std().detach()), 1.0, places=4)

    def test_standard_cnn_layerwise_wmf(self):
        model = StandardCNN(self.spec)
        report = initialise_model(
            model, self.loader, "di_wmf", self.device, layers=3
        )
        self.assertEqual(
            report.initialised_modules, ("conv1", "conv2", "classifier")
        )
        images, _ = next(iter(self.loader))
        logits = model(images)
        self.assertEqual(tuple(logits.shape), (5, 2))
        self.assertTrue(torch.isfinite(logits).all())
        anchors = capture_anchors(model, layers=3)
        self.assertEqual(set(anchors), {"conv1", "conv2"})
        self.assertEqual(model.anchor_classes("conv1").numel(), 8)

    def test_standard_cnn_exact_logit_decomposition_and_intervention(self):
        model = StandardCNN(self.spec)
        images, _ = next(iter(self.loader))
        decomposition = decompose_logits(model, images)
        self.assertTrue(
            torch.allclose(
                decomposition.logits,
                decomposition.reconstructed_logits,
                atol=1e-6,
            )
        )
        predictions = decomposition.logits.argmax(dim=1)
        scores = target_contributions(decomposition.contributions, predictions)
        selected = topk_mask(scores, 2)
        only_selected = intervened_logits(model, decomposition.features, selected)
        without_selected = intervened_logits(model, decomposition.features, ~selected)
        bias = model.classifier.bias[None, :]
        self.assertTrue(
            torch.allclose(
                only_selected + without_selected - bias,
                decomposition.logits,
                atol=1e-6,
            )
        )
        self.assertTrue(torch.equal(mask_jaccard(selected, selected), torch.ones(5)))

    def test_controlled_gabor_and_pca_stem_initialisers(self):
        for method in ("gabor", "pca"):
            model = StandardCNN(self.spec)
            report = initialise_model(
                model, self.loader, method, self.device, layers=1
            )
            self.assertEqual(report.initialised_modules, ("conv1",))
            self.assertTrue(torch.isfinite(model.conv1.weight).all())
            images, _ = next(iter(self.loader))
            self.assertTrue(torch.isfinite(model(images)).all())

    def test_lowrank_wmf_standard_initialiser(self):
        model = StandardCNN(self.spec)
        report = initialise_model(
            model,
            self.loader,
            "lowrank_wmf",
            self.device,
            layers=3,
            covariance_rank=4,
        )
        self.assertEqual(report.covariance, "diagonal_plus_lowrank")
        self.assertEqual(report.covariance_rank, 4)
        images, _ = next(iter(self.loader))
        self.assertTrue(torch.isfinite(model(images)).all())

    def test_adaptive_initialisation_uses_validation_only_candidates(self):
        loader = large_synthetic_loader()
        model, report, selection = select_adaptive_initialisation(
            lambda: StandardCNN(self.spec),
            loader,
            loader,
            self.device,
            [AdaptiveCandidate("kmeans"), AdaptiveCandidate("di_wmf")],
            layers=3,
            seed=3,
        )
        self.assertIn(selection.selected_method, {"kmeans", "di_wmf"})
        self.assertEqual(report.method, selection.selected_method)
        self.assertEqual(len(selection.candidates), 2)
        self.assertTrue(torch.isfinite(model(next(iter(loader))[0])).all())

    def test_shared_concept_and_residual_models(self):
        loader = large_synthetic_loader()
        concept = SharedConceptCNN(self.spec)
        report = initialise_model(
            concept, loader, "di_wmf", self.device, layers=3
        )
        self.assertEqual(report.covariance, "diagonal_shared")
        images, _ = next(iter(loader))
        self.assertEqual(tuple(concept(images).shape), (20, 2))
        self.assertGreaterEqual(concept.sparsity_penalty().item(), 0.0)
        with torch.no_grad():
            concept.classifier.weight.fill_(0.01)
        concept.proximal_sparsity_step(0.1)
        self.assertEqual(
            concept.sparsity_metrics()["fraction_exactly_zero"], 1.0
        )
        concept_decomposition = decompose_logits(concept, images)
        self.assertTrue(
            torch.allclose(
                concept_decomposition.logits,
                concept_decomposition.reconstructed_logits,
                atol=1e-5,
            )
        )
        residual = SmallResNet(self.spec)
        report = initialise_model(
            residual, loader, "di_wmf", self.device, layers=3
        )
        self.assertIn("classifier", report.initialised_modules)
        self.assertEqual(tuple(residual(images).shape), (20, 2))
        residual_decomposition = decompose_logits(residual, images)
        self.assertTrue(
            torch.allclose(
                residual_decomposition.logits,
                residual_decomposition.reconstructed_logits,
                atol=1e-5,
            )
        )

    def test_standard_resnet18_domain_initialisation(self):
        loader = large_synthetic_loader()
        model = StandardResNet18(self.spec)
        report = initialise_model(
            model, loader, "di_wmf", self.device, layers=3
        )
        self.assertEqual(report.initialised_modules, ("conv1", "classifier"))
        self.assertEqual(model.anchor_classes("conv1").numel(), 64)
        images, _ = next(iter(loader))
        logits = model(images)
        self.assertEqual(tuple(logits.shape), (20, 2))
        self.assertTrue(torch.isfinite(logits).all())
        decomposition = decompose_logits(model, images)
        self.assertTrue(
            torch.allclose(
                decomposition.logits,
                decomposition.reconstructed_logits,
                atol=1e-5,
            )
        )

    def test_correncoder_reconstruction_and_width_scaling(self):
        images, _ = next(iter(self.loader))
        correncoder = CorrEncoderCNN(self.spec)
        initialise_model(
            correncoder, self.loader, "lowrank_wmf", self.device,
            layers=3, covariance_rank=2,
        )
        reconstruction = correncoder.reconstruct(images)
        self.assertEqual(reconstruction.shape, images.shape)
        loss = correncoder.auxiliary_loss(images)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss.detach()), 0.0)
        self.assertLessEqual(float(loss.detach()), 2.0)

        full_correncoder = FullCorrEncoderCNN(self.spec)
        initialise_model(
            full_correncoder, self.loader, "lowrank_wmf", self.device,
            layers=3, covariance_rank=2,
        )
        full_correncoder.initialise_decoder_from_encoder()
        self.assertTrue(
            torch.allclose(
                full_correncoder.decoder1.weight,
                full_correncoder.conv1.weight,
            )
        )
        self.assertEqual(
            full_correncoder.reconstruct(images).shape,
            images.shape,
        )
        self.assertGreater(
            sum(p.numel() for p in full_correncoder.parameters()),
            sum(p.numel() for p in correncoder.parameters()),
        )

        full = StandardCNN(self.spec)
        half = HalfWidthCNN(self.spec)
        quarter = QuarterWidthCNN(self.spec)
        counts = [sum(p.numel() for p in item.parameters()) for item in (full, half, quarter)]
        self.assertGreater(counts[0], counts[1])
        self.assertGreater(counts[1], counts[2])
        for model in (half, quarter):
            report = initialise_model(
                model, self.loader, "di_wmf", self.device, layers=3
            )
            self.assertIn("classifier", report.initialised_modules)
            self.assertEqual(tuple(model(images).shape), (5, 2))

    def test_c4_model_has_exact_quarter_turn_invariance(self):
        model = C4RotationInvariantCNN(self.spec).eval()
        images, _ = next(iter(self.loader))
        reference = model(images)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            sum(parameter.numel() for parameter in StandardCNN(self.spec).parameters()),
        )
        for turns in (1, 2, 3):
            rotated = model(torch.rot90(images, turns, dims=(-2, -1)))
            self.assertTrue(torch.allclose(reference, rotated, atol=2e-5, rtol=1e-5))

    def test_published_correncoder_matches_reported_topology(self):
        model = PublishedCorrEncoder1D(dropout=0.5).eval()
        signal = torch.randn(3, 1, model.input_length)
        output = model(signal)
        self.assertEqual(output.shape, signal.shape)
        self.assertEqual(model.conv1.kernel_size, (150,))
        self.assertEqual(model.conv2.kernel_size, (75,))
        self.assertEqual(model.conv3.kernel_size, (50,))
        self.assertEqual(model.conv1.out_channels, 8)

    def test_correncoder_regression_losses_and_matched_initialisation(self):
        sampling_hz = 30
        time = torch.arange(PublishedCorrEncoder1D.input_length) / sampling_hz
        target = torch.sin(2 * torch.pi * 0.3 * time)
        targets = target.repeat(24, 1).unsqueeze(1)
        inputs = torch.roll(target, shifts=5).repeat(24, 1).unsqueeze(1)
        inputs = inputs + 0.02 * torch.randn_like(inputs)

        self.assertLess(float(pearson_correlation_loss(targets, targets)), 1e-6)
        self.assertGreater(float(pearson_correlation_loss(-targets, targets)), 1.9)
        constant = torch.zeros_like(targets[:2], requires_grad=True)
        constant_loss = pearson_correlation_loss(constant, targets[:2])
        constant_loss.backward()
        self.assertTrue(torch.isfinite(constant.grad).all())
        self.assertLess(float(spectral_shape_loss(targets, targets)), 1e-6)
        different_frequency = torch.sin(2 * torch.pi * 0.6 * time)
        different_frequency = different_frequency.repeat(24, 1).unsqueeze(1)
        dft_loss = spectral_shape_loss(different_frequency, targets)
        self.assertGreater(float(dft_loss), 0.01)
        window = torch.hann_window(target.shape[-1], periodic=False)
        frequencies = torch.fft.rfftfreq(target.shape[-1], d=1 / sampling_hz)
        band = (frequencies >= 0.08) & (frequencies <= 0.8)
        reference_spectra = []
        for signal in (different_frequency, targets):
            centred = signal - signal.mean(dim=-1, keepdim=True)
            spectrum = torch.fft.rfft(centred * window, dim=-1).abs()[..., band]
            reference_spectra.append(spectrum / spectrum.sum(dim=-1, keepdim=True))
        reference_loss = F.l1_loss(*reference_spectra)
        self.assertAlmostEqual(float(dft_loss), float(reference_loss), places=6)
        shifted = torch.roll(targets, shifts=9, dims=-1)
        zero_lag_loss = pearson_correlation_loss(shifted, targets)
        tolerant_loss = lag_tolerant_pearson_loss(
            shifted, targets, max_lag_samples=15, lag_step=3
        )
        soft_tolerant_loss = soft_lag_pearson_loss(
            shifted,
            targets,
            max_lag_samples=15,
            lag_step=3,
            temperature=0.05,
        )
        self.assertLess(float(tolerant_loss), float(zero_lag_loss))
        self.assertLess(float(soft_tolerant_loss), float(zero_lag_loss))
        soft_constant = torch.zeros_like(targets[:2], requires_grad=True)
        soft_constant_loss = soft_lag_pearson_loss(
            soft_constant, targets[:2], max_lag_samples=15, lag_step=3
        )
        soft_constant_loss.backward()
        self.assertTrue(torch.isfinite(soft_constant.grad).all())
        constant_correlation, constant_lag = lag_robust_waveform_correlation(
            np.zeros(120), np.sin(np.linspace(0.0, 4.0 * np.pi, 120)), 10, 2
        )
        self.assertEqual(constant_correlation, 0.0)
        self.assertEqual(constant_lag, 0)

        model = PublishedCorrEncoder1D(dropout=0.0)
        report = initialise_1d_matched_filter(
            model, inputs, targets, max_patches=2_000, seed=0
        )
        self.assertEqual(report.filters, 8)
        self.assertEqual(report.patches_used, 2_000)
        self.assertTrue(torch.isfinite(model.conv1.weight).all())
        self.assertTrue(torch.allclose(model.conv1.weight, model.decoder1.weight))
        output = model(inputs[:2])
        objective, components = correncoder_regression_loss(
            output,
            targets[:2],
            correlation_lambda=0.1,
            spectral_lambda=0.1,
        )
        self.assertTrue(torch.isfinite(objective))
        self.assertEqual(set(components), {"mse", "correlation_loss", "spectral_loss"})

        calibrated_model = PublishedCorrEncoder1D(dropout=0.0)
        initialise_1d_matched_filter(
            calibrated_model, inputs, targets, max_patches=500, seed=1
        )
        calibration = calibrate_regression_output(
            calibrated_model, inputs, targets, batch_size=8
        )
        self.assertLessEqual(calibration.train_mse_after, calibration.train_mse_before)

        safe_model = PublishedCorrEncoder1D(dropout=0.0)
        initialise_1d_matched_filter(safe_model, inputs, targets, max_patches=500, seed=1)
        safe_report = fit_safe_calibration_head(
            safe_model,
            inputs,
            targets,
            batch_size=8,
            minimum_absolute_gain=0.05,
        )
        self.assertAlmostEqual(
            safe_report.applied_initial_gain,
            safe_report.least_squares_gain,
            places=5,
        )
        self.assertEqual(safe_report.minimum_backward_gain, 0.05)
        calibration_head = SafeAffineCalibration(
            gain=0.001,
            offset=0.0,
            minimum_absolute_gain=0.05,
        )
        calibration_input = torch.ones(3, requires_grad=True)
        calibration_head(calibration_input).sum().backward()
        self.assertTrue(
            torch.allclose(calibration_input.grad, torch.full((3,), 0.05))
        )
        safe_output = safe_model(inputs[:2])
        safe_output.square().mean().backward()
        self.assertTrue(torch.isfinite(safe_model.conv1.weight.grad).all())
        self.assertGreater(
            abs(float(safe_model.output_calibration.gain_parameter.grad)), 0.0
        )

        layerwise_model = PublishedCorrEncoder1D(dropout=0.0)
        layerwise_report = initialise_1d_layerwise_matched_filter(
            layerwise_model,
            inputs,
            targets,
            stem_max_patches=500,
            deep_max_patches=500,
            seed=2,
        )
        self.assertEqual(set(layerwise_report["layers"]), {"conv1", "conv2", "conv3"})
        self.assertTrue(torch.allclose(layerwise_model.conv1.weight, layerwise_model.decoder1.weight))
        self.assertTrue(torch.allclose(layerwise_model.conv2.weight, layerwise_model.decoder2.weight))
        self.assertTrue(torch.allclose(layerwise_model.conv3.weight, layerwise_model.decoder3.weight))

        depth2_model = PublishedCorrEncoder1D(dropout=0.0)
        original_conv3 = depth2_model.conv3.weight.detach().clone()
        depth2_report = initialise_1d_layerwise_matched_filter(
            depth2_model,
            inputs,
            targets,
            stem_max_patches=500,
            deep_max_patches=500,
            depth=2,
            seed=3,
        )
        self.assertEqual(set(depth2_report["layers"]), {"conv1", "conv2"})
        self.assertTrue(torch.allclose(depth2_model.conv3.weight, original_conv3))

        lowrank_model = PublishedCorrEncoder1D(dropout=0.0)
        lowrank_report = initialise_1d_layerwise_matched_filter(
            lowrank_model,
            inputs,
            targets,
            stem_max_patches=500,
            deep_max_patches=500,
            deep_covariance="lowrank",
            covariance_rank=4,
            seed=4,
        )
        self.assertEqual(lowrank_report["covariance_rank"], 4)
        self.assertEqual(lowrank_report["layers"]["conv2"]["covariance_rank"], 4)
        self.assertTrue(torch.isfinite(lowrank_model.conv3.weight).all())

    def test_global_pruning_reaches_requested_sparsity(self):
        model = StandardCNN(self.spec)
        total, nonzero = apply_global_pruning(model, 0.5)
        self.assertEqual(total - nonzero, round(0.5 * total))
        images, _ = next(iter(self.loader))
        self.assertTrue(torch.isfinite(model(images)).all())

    def test_structured_pruning_physically_reduces_channels(self):
        model = StandardCNN(self.spec)
        pruned, stem_channels, body_channels = physically_prune(model, 0.5)
        self.assertEqual(stem_channels, model.conv1.out_channels // 2)
        self.assertEqual(body_channels, model.conv2.out_channels // 2)
        self.assertLess(
            sum(parameter.numel() for parameter in pruned.parameters()),
            sum(parameter.numel() for parameter in model.parameters()),
        )
        images, _ = next(iter(self.loader))
        self.assertEqual(pruned(images).shape, model(images).shape)

    def test_reliability_metrics_are_bounded(self):
        self.assertEqual(binary_auc(torch.tensor([0.9]).numpy(), torch.tensor([0.1]).numpy()), 1.0)
        logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [2.0, 1.0]])
        labels = torch.tensor([0, 1, 1])
        metrics = classification_metrics(logits, labels)
        self.assertGreaterEqual(metrics["ece15"], 0.0)
        self.assertLessEqual(metrics["ece15"], 1.0)
        self.assertGreaterEqual(metrics["aurc"], 0.0)


if __name__ == "__main__":
    unittest.main()
