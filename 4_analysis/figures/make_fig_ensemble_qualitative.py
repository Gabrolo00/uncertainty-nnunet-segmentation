"""
Figura qualitativa deep contro checkpoint ensemble sul rene: errori e mappe di entropia, MI e DEU su una fetta.

Uso: python make_fig_ensemble_qualitative.py <caso> [fetta] [out.png]
"""
import sys, nibabel as nib, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage as ndi
c=sys.argv[1]; z_arg=int(sys.argv[2]) if len(sys.argv)>2 else None
out=sys.argv[3] if len(sys.argv)>3 else f'draft_{c}.png'
R='/mnt/sda2/grolo/nnUNet/raw/Dataset904_KidneyTSMechanical'
P='/mnt/sda2/grolo/kidney/predictions'
U='/mnt/sda2/grolo'
ld=lambda p: np.asarray(nib.load(p).dataobj)
ct=ld(f'{R}/imagesTs/{c}_0000.nii.gz').astype(np.float32)
gt=ld(f'{R}/labelsTs/{c}.nii.gz')>0
ens={'Deep ensemble':(f'{P}/deep/{c}.nii.gz',f'{U}/kidney/deep/kidney_uncertainty_maps/{c}_'),
     'Checkpoint ensemble':(f'{P}/checkpoint/{c}.nii.gz',f'{U}/kidney/checkpoint/kidney_cyclr_uncertainty_maps/{c}_')}
maps=[('entropy','Entropy'),('mutual_information','Mutual information'),('DEU_entropy','DEU')]
D={}
for k,(pp,up) in ens.items():
    D[k]={'pred':ld(pp)>0}
    for m,_ in maps: D[k][m]=np.asarray(nib.load(up+m+'.nii.gz').get_fdata(),dtype=np.float32)
err=sum(((D[k]['pred']^gt)).sum(axis=(0,1)) for k in D)
z=z_arg if z_arg is not None else int(np.argmax(err))
print('slice',z,'errors per ens',[int((D[k]['pred']^gt)[:,:,z].sum()) for k in D])
sl=lambda a: np.fliplr(np.rot90(a[:,:,z]))   # RAS -> radiological convention (patient right on image left)
roi=sl(ndi.binary_dilation(gt|D['Deep ensemble']['pred']|D['Checkpoint ensemble']['pred'],iterations=10))
errz=sl(gt^D['Checkpoint ensemble']['pred'])
lab,_=ndi.label(roi); k_id=np.bincount(lab[errz].ravel()).argmax() if errz.any() else 1
ys,xs=np.where(lab==k_id); cy,cx=(ys.min()+ys.max())//2,(xs.min()+xs.max())//2
half=max(ys.max()-ys.min(),xs.max()-xs.min())//2+2
y0,y1,x0,x1=cy-half,cy+half,cx-half,cx+half
cr=lambda a: sl(a)[y0:y1,x0:x1]
ctw=np.clip((cr(ct)+160)/400,0,1)  # window W400 L40
vmax={m:max(np.percentile(cr(D[k][m]),99.5) for k in D) for m,_ in maps}
vmax['DEU_entropy']=vmax['entropy']=max(vmax['entropy'],vmax['DEU_entropy'])
h,w=y1-y0,x1-x0
fig=plt.figure(figsize=(8.2,4.6))
gs=fig.add_gridspec(3,4,height_ratios=[1,1,0.06],hspace=0.06,wspace=0.04,left=0.05,right=0.99,top=0.93,bottom=0.10)
ax=np.array([[fig.add_subplot(gs[r,cc]) for cc in range(4)] for r in range(2)])
cols=['Errors']+[t for _,t in maps]
for i,k in enumerate(D):
    p=cr(D[k]['pred']); g=cr(gt)
    a=ax[i,0]; a.imshow(ctw,cmap='gray',vmin=0,vmax=1)
    e=np.zeros(g.shape+(4,)); e[p&~g]=[1,0.55,0,1]; e[g&~p]=[0.1,0.6,1,1]; a.imshow(e)
    a.contour(g,[0.5],colors=['#00c000'],linewidths=1.0); a.contour(p,[0.5],colors=['#ff3030'],linewidths=0.9)
    for j,(m,_) in enumerate(maps):
        a=ax[i,1+j]; im=a.imshow(cr(D[k][m]),cmap='magma',vmin=0,vmax=vmax[m])
        a.contour(g,[0.5],colors=['#00c000'],linewidths=0.6)
        if i==1: fig.colorbar(im,cax=fig.add_subplot(gs[2,1+j]),orientation='horizontal')
    ax[i,0].set_ylabel(k.split()[0],fontsize=11)
for j,t in enumerate(cols): ax[0,j].set_title(t,fontsize=11)
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([],[],color='#00c000',label='Reference'),Line2D([],[],color='#ff3030',label='Prediction'),
   Patch(color=(1,0.55,0),label='False positive'),Patch(color=(0.1,0.6,1),label='False negative')],
   loc='lower left',ncol=2,fontsize=8,columnspacing=0.8,handlelength=1.4,frameon=False,bbox_to_anchor=(0.0,0.0))
fig.savefig(out,dpi=200)
