# Notebook-derived stage: ATML_PA1_Task1.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 17
import ipywidgets as widgets
from IPython.display import clear_output

review = pd.read_csv(review_file, dtype={'accept': 'string'}, keep_default_na=False)
STRATA = [pair for a, b in PAIRS for pair in [(a, b), (b, a)]]

review_out = widgets.Output()
accept_btn = widgets.Button(description='Accept', button_style='success')
reject_btn = widgets.Button(description='Reject', button_style='danger')

def next_candidate():
    for content_class, style_class in STRATA:
        group = review[
            (review.content_class == content_class)
            & (review.style_class == style_class)
        ]
        if (group['accept'] == '1').sum() >= 20:
            continue

        remaining = group.index[group['accept'] == '']
        if not len(remaining):
            return None, f'Fewer than 20 valid conflicts for {content_class} → {style_class}.'
        return int(remaining[0]), None

    return None, None

def show_next_candidate():
    k, error = next_candidate()
    with review_out:
        clear_output(wait=True)
        if error:
            accept_btn.disabled = reject_btn.disabled = True
            print(error)
            return
        if k is None:
            accept_btn.disabled = reject_btn.disabled = True
            print('Review complete. Run the validation cell below.')
            return

        m = review.iloc[k]
        print(
            f"{int((review['accept'] == '1').sum())}/200 accepted · "
            f"{int((review['accept'] == '0').sum())} rejected · "
            f"candidate {k + 1}/400"
        )
        print(f'Content: {m.content_class} | Style: {m.style_class}')

        canvas = Image.new('RGB', (3 * 224, 224), 'white')
        for j, img in enumerate(
            [X[int(m.content_id)], X[int(m.style_id)], CAND[k]]
        ):
            canvas.paste(Image.fromarray(img), (j * 224, 0))

        display(canvas)
        print('Left: content | Middle: style | Right: conflict')

def record_review(value):
    k, _ = next_candidate()
    if k is None:
        return
    review.loc[k, 'accept'] = value
    review.to_csv(review_file, index=False)
    show_next_candidate()

accept_btn.on_click(lambda _: record_review('1'))
reject_btn.on_click(lambda _: record_review('0'))
display(widgets.HBox([accept_btn, reject_btn]), review_out)
show_next_candidate()
