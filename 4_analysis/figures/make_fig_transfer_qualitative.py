"""
Figura qualitativa from scratch contro fine-tuned sul test esterno: errori e mappe di mutual information.

Uso: python make_fig_transfer_qualitative.py <caso1,caso2,...> <out.png>
"""
import sys, nibabel as nib, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
B='/mnt/sda2/grolo/kidney/external/'
cases=sys.argv[1].split(','); out=sys.argv[2]
ld=lambda p: np.asarray(nib.load(B+p).dataobj)
fig=plt.figure(figsize=(6.6,2.15*len(cases)+0.45))
gs=fig.add_gridspec(len(cases),4,width_ratios=[1,1,1,0.06],hspace=0.06,wspace=0.04,left=0.06,right=0.91,top=0.93,bottom=0.55/(2.15*len(cases)+0.45)+0.01)
for r,c in enumerate(cases):
    ct=ld(f'cropped/images/{c}_0000.nii.gz').astype(np.float32); gt=ld(f'cropped/labelsTs/{c}.nii.gz')>0
    pr=ld(f'pred_deep/{c}.nii.gz')>0
    mi={k:np.asarray(nib.load(B+f'maps_{k}/{c}_mutual_information.nii.gz').get_fdata(),dtype=np.float32) for k in ['deep','finetune']}
    err=(gt^pr)
    z=int(np.argmax((gt&~pr).sum(axis=(0,1))))
    sl=lambda a: a[:,:,z].T   # LPS: anterior on top, patient left on image right (radiological)
    roi=ndi.binary_dilation(sl(gt|pr),iterations=8); lab,_=ndi.label(roi)
    kid=np.bincount(lab[sl(err)].ravel()).argmax()
    ys,xs=np.where(lab==kid); cy,cx=(ys.min()+ys.max())//2,(xs.min()+xs.max())//2
    half=max(ys.max()-ys.min(),xs.max()-xs.min())//2+2
    cr=lambda a: sl(a)[cy-half:cy+half,cx-half:cx+half]
    ctw=np.clip((cr(ct)+160)/400,0,1); g=cr(gt); p=cr(pr)
    vmax=max(np.percentile(cr(mi[k]),99.5) for k in mi)
    a=fig.add_subplot(gs[r,0]); a.imshow(ctw,cmap='gray',vmin=0,vmax=1)
    e=np.zeros(g.shape+(4,)); e[p&~g]=[1,0.55,0,1]; e[g&~p]=[0.1,0.6,1,1]; a.imshow(e)
    a.contour(g,[0.5],colors=['#00c000'],linewidths=1.0); a.contour(p,[0.5],colors=['#ff3030'],linewidths=0.9)
    axs=[a]
    if len(cases)>1: a.set_ylabel(f'Case {chr(65+r)}',fontsize=10)
    for j,k in enumerate(['deep','finetune']):
        a=fig.add_subplot(gs[r,1+j]); im=a.imshow(cr(mi[k]),cmap='magma',vmin=0,vmax=vmax)
        a.contour(g,[0.5],colors=['#00c000'],linewidths=0.6); axs.append(a)
    fig.colorbar(im,cax=fig.add_subplot(gs[r,3]))
    for a in axs: a.set_xticks([]); a.set_yticks([])
    if r==0:
        for a,t in zip(axs,['Errors (from scratch)','MI, from scratch','MI, fine-tuned']): a.set_title(t,fontsize=10)
    print(c,'slice',z,'vmax',vmax,'cx',cx,'side','left' if cx>256 else 'right')
fig.legend(handles=[Line2D([],[],color='#00c000',label='Reference'),Line2D([],[],color='#ff3030',label='Prediction'),
   Patch(color=(1,0.55,0),label='False positive'),Patch(color=(0.1,0.6,1),label='False negative')],
   loc='lower center',ncol=4,fontsize=8,frameon=False,bbox_to_anchor=(0.5,0.0))
fig.savefig(out,dpi=200)
