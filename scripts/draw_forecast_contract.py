"""Vector schematic of the inference contract; contains no experimental data."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'paper/figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(9.4,2.9));ax.set(xlim=(0,10),ylim=(0,3));ax.axis('off')
    def box(x,y,w,h,text,color='#E8F2F1'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06,rounding_size=0.08',facecolor=color,edgecolor='#34625E',linewidth=1))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center')
    def arrow(a,b,dashed=False):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,color='#444444',linewidth=1,linestyle='--' if dashed else '-'))
    box(.1,1.05,1.55,.75,'Observed events\nat or before t')
    box(2,1.9,1.75,.75,'Individual CDFs\nF₁, …, Fₙ')
    box(2,.2,1.75,.75,'Current cohort\nand typed relations')
    box(4.2,1.05,1.6,.75,'Joint sampler\nshared + private factors')
    box(6.25,1.05,1.55,.75,'One exit bin\nper entity and draw')
    box(8.2,1.05,1.65,.75,'Group projections\nand forecast scores')
    box(8.2,.15,1.65,.5,'Held-out outcomes','#F5EFE5')
    arrow((1.65,1.55),(2,2.15));arrow((1.65,1.25),(2,.7));arrow((3.75,2.2),(4.2,1.6));arrow((3.75,.65),(4.2,1.25));arrow((5.8,1.425),(6.25,1.425));arrow((7.8,1.425),(8.2,1.425));arrow((9.025,.65),(9.025,1.05),True)
    ax.text(9.6,.85,'score only',ha='center',va='center',fontsize=8)
    ax.text(5.0,2.85,'Identical individual CDFs across dependence controls',ha='center',fontsize=9,color='#345F5B')
    fig.tight_layout(pad=.2);fig.savefig(out/'forecast_contract.pdf');plt.close(fig)


if __name__=='__main__':main()
