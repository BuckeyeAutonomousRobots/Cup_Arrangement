"""Render existing STEP files with per-pixel depth testing (no painter sorting)."""
from pathlib import Path
import cadquery as cq
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
BACKGROUND = np.array([244,246,248], dtype=float)/255


def render_panel(items, span, target, size=1100):
    """Orthographic triangle rasterizer with a shared z-buffer for all parts."""
    az, el = np.deg2rad([-62,29])
    toward = np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
    right = np.array([-np.sin(az),np.cos(az),0])
    up = np.cross(toward,right)
    basis = np.stack([right,up,toward],axis=1)
    pixels = np.broadcast_to(BACKGROUND,(size,size,3)).copy()
    depth = np.full((size,size),-np.inf)
    light = np.array([.25,-.35,.9027735])
    for shape,color in items:
        vertices,triangles=shape.tessellate(.02,.025)
        world=np.array([v.toTuple() for v in vertices])
        projected=(world-np.asarray(target))@basis
        projected[:,:2]*=size/span
        projected[:,0]+=size/2
        projected[:,1]=size/2-projected[:,1]
        for face in triangles:
            a,b,c=projected[list(face)]
            x0=max(0,int(np.floor(min(a[0],b[0],c[0]))))
            x1=min(size-1,int(np.ceil(max(a[0],b[0],c[0]))))
            y0=max(0,int(np.floor(min(a[1],b[1],c[1]))))
            y1=min(size-1,int(np.ceil(max(a[1],b[1],c[1]))))
            if x1<x0 or y1<y0: continue
            denom=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
            if abs(denom)<1e-12: continue
            yy,xx=np.mgrid[y0:y1+1,x0:x1+1]
            xx=xx+.5;yy=yy+.5
            wa=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/denom
            wb=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/denom
            wc=1-wa-wb
            z=wa*a[2]+wb*b[2]+wc*c[2]
            region=depth[y0:y1+1,x0:x1+1]
            mask=(wa>=-1e-9)&(wb>=-1e-9)&(wc>=-1e-9)&(z>region)
            normal=np.cross(world[face[1]]-world[face[0]],world[face[2]]-world[face[0]])
            normal/=max(np.linalg.norm(normal),1e-12)
            brightness=.64+.36*abs(normal@light)
            region[mask]=z[mask]
            pixels[y0:y1+1,x0:x1+1][mask]=np.asarray(color)*brightness
    return pixels


def render():
    cup=cq.importers.importStep(str(OUT/'cup.step')).val()
    saucer=cq.importers.importStep(str(OUT/'saucer.step')).val()
    cream=(.95,.92,.85);blue=(.45,.72,.95)
    fig,axes=plt.subplots(1,3,figsize=(15,6),facecolor='#f4f6f8')
    fig.suptitle('Cup & saucer | circular CAD',fontsize=22,color='#243447',y=.97)
    panels=[
        ('CUP  /  Ø45 mm body × 50 mm',[(cup,cream)],90,(7,0,25)),
        ('SAUCER  /  Ø110 × 8 mm',[(saucer,blue)],140,(0,0,4)),
        ('ASSEMBLED  /  centered preview',[(saucer,blue),(cup.translate((0,0,8)),cream)],140,(0,0,25)),
    ]
    for ax,(title,items,span,target) in zip(axes,panels):
        ax.imshow(render_panel(items,span,target),interpolation='lanczos')
        ax.set_axis_off();ax.set_title(title,fontsize=11,color='#243447')
    fig.text(.5,.06,'True circular body • 3 mm cup walls/base • rectangular handle • flat cylindrical saucer',ha='center',fontsize=12,color='#526476')
    fig.subplots_adjust(left=.025,right=.975,bottom=.13,top=.84,wspace=.04)
    fig.savefig(OUT/'preview.png',dpi=160,facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == '__main__':
    render()
