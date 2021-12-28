## Vision Transformer for Small-Size Datasets

**Seung Hoon Lee and Seunghyun Lee and Byung Cheol Song** | [Paper](https://arxiv.org/abs/2112.13492)

Inha University

### Abstract
Recently, the Vision Transformer (ViT), which applied the transformer structure to the image classification task, has outperformed convolutional neural networks. However, the high performance of the ViT results from pre-training using a large-size dataset such as JFT-300M, and its dependence on a large dataset is interpreted as due to low locality inductive bias. This paper proposes Shifted Patch Tokenization (SPT) and Locality Self-Attention (LSA), which effectively solve the lack of locality inductive bias and enable it to learn from scratch even on small-size datasets. Moreover, SPT and LSA are generic and effective add-on modules that are easily applicable to various ViTs. Experimental results show that when both SPT and LSA were applied to the ViTs, the performance improved by an average of 2.96% in Tiny-ImageNet, which is a representative small-size dataset. Especially, Swin Transformer achieved an overwhelming performance improvement of 4.08% thanks to the proposed SPT and LSA.

<div align="center">
  <img src="SPT.png" width="50%" title="" alt="teaser">
  **<Shifted Patch Tokenization>
  </img><img src="LSA.png" width="50%" title="" alt="teaser"></img>
  <img src="main.png" width="50%" title="" alt="teaser"></img>
</div>

## Citation

```
@misc{lee2021vision,
      title={Vision Transformer for Small-Size Datasets}, 
      author={Seung Hoon Lee and Seunghyun Lee and Byung Cheol Song},
      year={2021},
      eprint={2112.13492},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
