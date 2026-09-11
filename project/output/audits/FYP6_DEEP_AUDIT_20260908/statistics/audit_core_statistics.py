"""Read-only audit of controls, seeds, downstream RNG and cost estimands.

Run with the recorded CL2 environment from the project root. This creates only
audit outputs, never modifies original experiment files and performs no training.
"""
from __future__ import annotations
import argparse, csv, itertools, json, math, sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from scipy.stats import t, ttest_rel


def rows(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def write(path,data):
    if not data:return
    fields=list(dict.fromkeys(k for r in data for k in r))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)

def holm(ps):
    out=np.ones(len(ps));run=0.
    for rank,index in enumerate(np.argsort(ps)):
        run=max(run,(len(ps)-rank)*ps[index]);out[index]=min(1.,run)
    return out.tolist()

def dump(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--skip-torch',action='store_true');a=p.parse_args()
    root=a.root.resolve();out=Path(__file__).resolve().parent;sys.path.insert(0,str(root))
    control_root=root/'outputs_convergence50_fitted_head_control'
    configs=[];main_index={};endpoints=[]
    for path in (root/'outputs_convergence50').glob('*/config.json'):
        c=json.loads(path.read_text());key=(c['dataset'],c['init'],int(c['seed']))
        if key in main_index:raise ValueError(f'Duplicate {key}')
        main_index[key]=(path.parent,c)
    for (d,m,s),(path,c) in main_index.items():
        hist=rows(path/'history.csv');metric=json.loads((path/'final_metrics.json').read_text())
        best=max(float(r['val_accuracy']) for r in hist)
        epoch=next(int(r['epoch']) for r in hist if float(r['val_accuracy'])==best)
        endpoints.append(dict(dataset=d,method=m,seed=s,epochs=c['epochs'],history_rows=len(hist),expected_best_epoch=epoch,recorded_best_epoch=metric['best_epoch'],matches=epoch==metric['best_epoch']))
    for d in ('fashion','cifar10','sign'):
        for s in range(10):
            ctrl_dir=control_root/f'{d}_standard_kaiming_conv_fitted_head_seed{s}'
            ctrl=json.loads((ctrl_dir/'control_config.json').read_text())
            _,di=main_index[d,'lowrank_wmf',s]
            mapping={'epochs':'epochs','batch_size':'batch_size','learning_rate':'learning_rate','train_fraction':'train_fraction','validation_fraction':'val_fraction','scheduler':'lr_scheduler','minimum_learning_rate':'min_learning_rate','covariance_rank':'covariance_rank','shrinkage':'shrinkage','device_name':'device_name'}
            for ck,dk in mapping.items():configs.append(dict(dataset=d,seed=s,field=ck,control=ctrl[ck],di=di[dk],matches=ctrl[ck]==di[dk]))
            configs.append(dict(dataset=d,seed=s,field='num_workers',control=ctrl.get('num_workers','NOT_RECORDED'),di=di.get('num_workers'),matches='UNVERIFIABLE_FROM_CONTROL_CONFIG'))
            hist=rows(ctrl_dir/'history.csv');best=max(float(r['val_accuracy']) for r in hist);epoch=next(int(r['epoch']) for r in hist if float(r['val_accuracy'])==best)
            endpoints.append(dict(dataset=d,method='kaiming_conv_fitted_head',seed=s,epochs=ctrl['epochs'],history_rows=len(hist),expected_best_epoch=epoch,recorded_best_epoch=ctrl['best_epoch'],matches=epoch==ctrl['best_epoch']))
    write(out/'fitted_control_settings.csv',configs);write(out/'best_validation_endpoint_checks.csv',endpoints)
    # Exact sign-flip is a sensitivity check under a symmetry null, not a claim
    # that the published Student-t test is mathematically wrong.
    raw=rows(root/'fitted_head_training_control_results/training_control_rows.csv')
    idx={(r['dataset'],r['condition'],int(r['seed'])):r for r in raw}
    assert len(idx)==len(raw)==90
    original=rows(root/'fitted_head_training_control_results/training_control_paired.csv');checks=[]
    for r in original:
        if r['contrast']=='head_fit_effect_on_kaiming_conv':left,right='kaiming_conv__fitted_head','kaiming_conv__kaiming_head'
        else:left,right='di_conv__fitted_head','kaiming_conv__fitted_head'
        d=r['dataset'];metric=r['metric'];lv=np.array([float(idx[d,left,s][metric]) for s in range(10)]);rv=np.array([float(idx[d,right,s][metric]) for s in range(10)]);delta=lv-rv
        mean=float(delta.mean());half=float(t.ppf(.975,9)*delta.std(ddof=1)/math.sqrt(10));praw=float(ttest_rel(lv,rv).pvalue)
        signs=np.array(list(itertools.product((-1,1),repeat=10)));permutations=np.abs((signs*delta).mean(1));pperm=float(np.mean(permutations>=abs(mean)-1e-12))
        checks.append(dict(dataset=d,contrast=r['contrast'],metric=metric,n=10,mean=mean,ci_low=mean-half,ci_high=mean+half,p_t=praw,p_exact_signflip=pperm,positive_seeds=int((delta>0).sum()),negative_seeds=int((delta<0).sum()),zero_seeds=int((delta==0).sum()),published_scope_p=float(r['p_value_holm_across_tasks']),max_stored_difference=max(abs(mean-float(r['mean_paired_difference'])),abs(mean-half-float(r['ci95_low'])),abs(mean+half-float(r['ci95_high']))),seed_differences=json.dumps(delta.tolist())))
    for r,q in zip(checks,holm([r['p_t'] for r in checks])):r['holm_all_18_sensitivity']=q
    for contrast in {r['contrast'] for r in checks}:
        subset=[r for r in checks if r['contrast']==contrast]
        for r,q in zip(subset,holm([r['p_t'] for r in subset])):r['holm_9_per_contrast_sensitivity']=q
    write(out/'fitted_control_inference_sensitivity.csv',checks)
    cost=rows(root/'total_cost_results/cost_rows.csv');energy=[];cost_replay=[]
    common_idle=float(json.loads((root/'total_cost_results/summary.json').read_text())['common_idle_watts_for_net_energy'])
    for path in sorted({Path(r['run_dir']) for r in cost}):
        c=json.loads((path/'config.json').read_text());m=json.loads((path/'final_metrics.json').read_text());e=m['energy_measurement'];tr=e['training_and_evaluation'];ini=e['initialisation']
        energy.append(dict(dataset=c['dataset'],method=c['init'],seed=c['seed'],training_seconds=m['training_seconds'],power_sampler_duration=tr['duration_seconds'],sampler_over_training_ratio=tr['duration_seconds']/m['training_seconds'],extra_sampled_seconds=tr['duration_seconds']-m['training_seconds'],power_samples=tr['sample_count'],mean_training_watts=tr['mean_watts'],initialisation_seconds=c['initialisation_seconds'],initialisation_sample_seconds=ini['duration_seconds'],init_power_samples=ini['sample_count'],run_dir=str(path)))
        hist=rows(path/'history.csv')
        for r in [r for r in cost if Path(r['run_dir'])==path]:
            hit=next((h for h in hist if float(h['val_accuracy'])>=float(r['threshold_accuracy'])),None)
            elapsed=float(hit['elapsed_seconds']) if hit else float(m['training_seconds'])
            fraction=min(max(elapsed/float(m['training_seconds']),0),1)
            estimate=max(0,ini['mean_watts']-common_idle)*ini['duration_seconds']+fraction*max(0,tr['mean_watts']-common_idle)*tr['duration_seconds']
            cost_replay.append(dict(dataset=c['dataset'],method=c['init'],seed=c['seed'],target=r['threshold_accuracy'],reached=hit is not None,epoch=int(hit['epoch']) if hit else '',training_fraction=fraction,recomputed_net_joules=estimate,stored_net_joules=float(r['total_net_joules_to_threshold_or_censor']),net_energy_difference=estimate-float(r['total_net_joules_to_threshold_or_censor']),time_difference=float(c['initialisation_seconds'])+elapsed-float(r['total_seconds_to_threshold_or_censor'])))
    write(out/'energy_time_window_mismatch.csv',energy)
    write(out/'cost_estimator_replay.csv',cost_replay)
    aggregate=rows(root/'total_cost_results/cost_aggregate.csv')
    write(out/'cost_incomplete_target_groups.csv',[r for r in aggregate if int(r['reached_runs'])<int(r['runs'])])
    summary={'formal_main_runs':len(main_index),'endpoint_checks':len(endpoints),'endpoint_mismatches':sum(not r['matches'] for r in endpoints),'settings_mismatches':[r for r in configs if r['matches'] is False],'worker_count_missing':sum(r['matches']=='UNVERIFIABLE_FROM_CONTROL_CONFIG' for r in configs),'core_contrasts_recomputed':len(checks),'max_inference_value_difference':max(r['max_stored_difference'] for r in checks),'energy_runs':len(energy),'energy_sampler_training_ratio_min':min(r['sampler_over_training_ratio'] for r in energy),'energy_sampler_training_ratio_max':max(r['sampler_over_training_ratio'] for r in energy),'cost_groups_with_censoring':sum(int(r['reached_runs'])<int(r['runs']) for r in aggregate)}
    if not a.skip_torch:
        import torch
        from run_experiment import seed_everything
        from domain_mf.models import build_model
        from domain_mf.initializers import initialise_model,calibrate_logit_scale
        from domain_mf.data import build_loaders
        device=torch.device('cuda');identity=[]
        for d in ('fashion','cifar10','sign'):
            for s in range(10):
                main_dir,_=main_index[d,'kaiming',s]
                control_dir=control_root/f'{d}_standard_kaiming_conv_fitted_head_seed{s}'
                ka=torch.load(main_dir/'filters_initial.pt',map_location='cpu',weights_only=True)
                co=torch.load(control_dir/'filters_initial.pt',map_location='cpu',weights_only=True)
                for layer in ('conv1','conv2'):
                    identity.append(dict(dataset=d,seed=s,layer=layer,equal=torch.equal(ka[layer],co[layer]),max_abs_diff=float((ka[layer]-co[layer]).abs().max())))
        write(out/'saved_kaiming_convolution_identity.csv',identity)
        summary['fitted_control_conv_layers']=len(identity);summary['fitted_control_conv_mismatches']=sum(not r['equal'] for r in identity)
        probe={};models={}
        for method in ('kaiming','random_stem','gabor'):
            path=next((root/'outputs_first_layer_cross_task').glob(f'fashion_standard_{method}_layers1_seed0_*/config.json'))
            c=json.loads(path.read_text());seed_everything(0)
            _,fit,_,_=build_loaders('fashion',Path(c['data_root']),Path(c['sign_root']),c['batch_size'],0,.1,.1,0)
            model=build_model('standard','fashion').to(device)
            initialise_model(model,fit,method,device,layers=1)
            before={k:v.detach().clone() for k,v in model.state_dict().items()}
            scale=calibrate_logit_scale(model,fit,device)
            saved=torch.load(path.parent/'filters_initial.pt',map_location=device,weights_only=True)
            probe[method]={'conv1_matches_archived_bank':torch.equal(saved['conv1'],model.conv1.weight),'conv1_max_difference':float((saved['conv1']-model.conv1.weight).abs().max()),'replayed_scale':scale,'recorded_scale':c['initialisation_report']['output_scale'],'covariance_rank_record':c['covariance_rank'],'batch_size':c['batch_size']}
            models[method]=before
        comparisons=[]
        for left,right in (('kaiming','random_stem'),('kaiming','gabor'),('random_stem','gabor')):
            for layer in ('conv2.weight','classifier.weight'):
                l=models[left][layer].flatten();r=models[right][layer].flatten()
                comparisons.append({'left':left,'right':right,'layer':layer,'equal':torch.equal(l,r),'max_abs_diff':float((l-r).abs().max()),'cosine':float(torch.nn.functional.cosine_similarity(l[None],r[None]))})
        dump(out/'first_layer_downstream_rng_probe.json',{'archived_source_checks':probe,'raw_later_layer_comparisons_before_output_calibration':comparisons,'torch_version':torch.__version__,'cuda':torch.version.cuda})
        # A diagnostic of the missing control num_workers metadata: this does
        # not assert the historical runs used different worker settings.
        orders={}
        for workers in (0,2):
            generator=torch.Generator().manual_seed(0)
            loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.arange(20)),batch_size=5,shuffle=True,generator=generator,num_workers=workers,persistent_workers=workers>0)
            orders[str(workers)]=[torch.cat([batch[0] for batch in loader]).tolist() for _ in range(3)]
        dump(out/'dataloader_worker_order_sensitivity.json',{'role':'counterexample showing why worker persistence is relevant metadata; not evidence of different historical settings','epoch_order_same':[orders['0'][e]==orders['2'][e] for e in range(3)],'orders':orders})
    summary['cost_row_checks']=len(cost_replay);summary['max_cost_energy_replay_difference']=max(abs(r['net_energy_difference']) for r in cost_replay);summary['max_cost_time_replay_difference']=max(abs(r['time_difference']) for r in cost_replay)
    dump(out/'summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
