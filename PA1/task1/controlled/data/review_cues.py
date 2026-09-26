# Notebook-derived stage: ATML_PA1_Task1_Controlled.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 18

import ipywidgets as widgets
from IPython.display import clear_output
review=pd.read_csv(review_file,dtype={'accept':'string'},keep_default_na=False)
STRATA=[pair for a,b in PAIRS for pair in [(a,b),(b,a)]]
review_out=widgets.Output()
accept_btn=widgets.Button(description='Accept',button_style='success')
reject_btn=widgets.Button(description='Reject',button_style='danger')

def next_candidate():
    for content_class,style_class in STRATA:
        group=review[(review.content_class==content_class)&(review.style_class==style_class)]
        if (group['accept']=='1').sum()>=20: continue
        remaining=group.index[group['accept']=='']
        if not len(remaining):
            return None,f'Insufficient valid conflicts for {content_class} → {style_class}; run the expansion cell and continue blinded review.'
        return int(remaining[0]),None
    return None,None

def show_next_candidate():
    k,error=next_candidate()
    with review_out:
        clear_output(wait=True)
        if error:
            accept_btn.disabled=reject_btn.disabled=True
            print(error); return
        if k is None:
            accept_btn.disabled=reject_btn.disabled=True
            print('Review complete: 20 accepted in each of 10 directions. Run the validation cell.')
            return
        m=review.iloc[k]
        accepted_count=int((review['accept']=='1').sum())
        rejected_count=int((review['accept']=='0').sum())
        print(f'{accepted_count}/200 accepted · {rejected_count} rejected · candidate {k+1}/{len(CAND)}')
        print(f'Content: {m.content_class} | Style: {m.style_class}')
        canvas=Image.new('RGB',(4*224,224),'white')
        isolated=(X[int(m.content_id)].astype(np.float32)*(MASKS[k,:,:,None]/255)+255*(1-MASKS[k,:,:,None]/255)).round().astype(np.uint8)
        for j,img in enumerate([X[int(m.content_id)],isolated,texture_tile(X[int(m.style_id)]),CAND[k]]):
            canvas.paste(Image.fromarray(img),(j*224,0))
        display(canvas)
        print('Original | Isolated content | Tiled style | Final conflict')

def record_review(value):
    k,error=next_candidate()
    if k is None: return
    review.loc[k,'accept']=value
    review.to_csv(review_file,index=False)
    show_next_candidate()

accept_btn.on_click(lambda _:record_review('1'))
reject_btn.on_click(lambda _:record_review('0'))
display(widgets.HBox([accept_btn,reject_btn]),review_out)
show_next_candidate()
