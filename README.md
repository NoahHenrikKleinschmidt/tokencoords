# TokenCoords

> TLDR:
> Turns coordinates of annotations within images into a tokenized form that can be passed to vision-language-models

Vision-Language Models (VLMs) combine the power of computer vision and natural language processing, by passing a tokenized form of an image, together with additional textual information to powerful large language models, producing textual output in the end. Google have shown that their VLM [Paligemma](https://ai.google.dev/gemma/docs/paligemma) is able to understand various vision tasks and solve problems of object detection or segmentation (see their [paper](https://arxiv.org/abs/2407.07726)). 

However, in order to let Paligemma's "brain", the language model Gemma-3B, undertand various aspects of computer vision tasks, it requires a textual input. In their case they use 1024 "location tokens" `<locXXXX>` that are added to the GemmaTokenizer's vocabulary to represent coordinates along each spatial axis of an image, thereby turning a point from `(x, y)` to  `<locAAAA><locBBBB>`. 

While the idea is simple and straightforward to implement, I felt it was more useful to have it stored in one accessible location to reuse in my projects. 

### Example Usage 1

Assuming we are fine-tuning the pretrained `paligemma-3b-224` checkpoint that is available on [Huggingface](https://huggingface.co/google/paligemma-3b-pt-224), but need to make our own dataset for it that includes bounding box annotations. 

We can tokenize the bounding box data like so:

```python
from tokencoords.paligemma import bbox_to_tokens, tokens_to_bbox

# bounding box coordinates of format
# y_topleft, x_topleft, y_bottomright, x_bottomright
bbox = [5, 5, 20, 20]


# turn the coordinates into tokens
tokenized = bbox_to_tokens(bbox, image_shape=(224, 224))
print(tokenized)
# ['<loc0021><loc0021><loc0090><loc0090>']

# convert back to coordinates
bbox_rev = tokens_to_bbox(tokenized, image_shape=(224, 224))
bbox_rev = bbox_rev.round()
print(bbox_rev)
# [[ 5.  5. 20. 20.]]
```
