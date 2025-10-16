# WISP-Bench

WISP-Bench consists of a three-tiered set of pass-or-fail unit-tests, each of which asks: *Given the ground truth image of the poem, does the linearized text accurately capture a specific whitespace property?*  This design was inspired by [olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench), and the unit test guidelines are in the appendix of our paper.

We retrieve ground truth images of relevant poems by converting Poetry Foundation webpages to images. For copyright reasons, we can share neither the original webpages nor the screenshotted images, but we do share our code to generate them with the `poetry-screenshotter` tool in this repo. Please see the `2-poetry-screenshotter/README.md` for more details.

## Prodigy Recipes for WISP-Bench

We use [Prodigy](https://prodi.gy/) as our annotation tool for the manual two-task benchmarking of linearization methods. Unfortunately, it is a one-time license purchase, so we cannot share the tool itself. However, we can share the Prodigy recipes (Python code) that we used to set up the annotation tasks.

### Task I: Label-Balanced Classification and Selection of 76 Poems
It is not possible to annotate all of the 19k poems in our original dataset; one of the challenges of this task is that it can't be reliably automated, either. Hence, we needed to select a much smaller subset to enable feasible manual annotation. At the same time, we also need to ensure that the selected subset maximizes presence and distribution of different WISP labels to avoid a sample size skewed towards a few dominant labels. In other words, we must get the most out of each manual annotation. 

To achieve this, we first set up the annotation task `wisp-label-classification.py` over 1000 randomly selected poems, where we manually labeled each poem's image with all applicable WISP labels. The annotation task produces a JSONL file where each line is an annotation; we run the script `wisp_mapping.py` to extract the mapping of WISP labels to poem ids from these annotations. Finally, we run a selection script `shortlist.py` that picks a shortlist of poems such that each WISP label l is represented by at least l_i (a predefined number) of poems, and the overall distribution of labels is as balanced as possible.

```
$ prodigy wisp.classify wisp_multilabel ./poem_images -F wisp-detection/image_classification.py --remove-base64  # set up proidgy annotation server and annotate
$ prodigy db-out wisp_multilabel > wisp_classifications.jsonl  # once done, export annotations to a JSONL file
$ python3 wisp_mapping.py wisp_classifications.jsonl # extract mapping of WISP labels to poem ids
$ python3 shortlist.py -l 20 -v 20 -p 20 -i 20  # generate a shortlist of poems with at least 20 poems per label
```

For our benchmark, we set the minimum number of poems per label at 20, resulting in a shortlist of 76 poems. We cannot share the poems themselves due to copyright restrictions; please contact the authors to request access to the shortlist of poems if used for research purposes.

### Task II: WISP-Bench 

For each of




