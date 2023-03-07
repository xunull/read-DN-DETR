# DN-DETR

原始项目地址：https://github.com/IDEA-Research/DN-DETR

论文中提到过的两个模型；
1. DN-DETR  对应于models中的DN_DAB_DETR
2. DN-Deformable-DETR  对应于models中的dn_dab_deformable_detr


## DN-DETR

1. attention.py
2. backbone.py
3. matcher.py
4. position_encoding.py
5. segmentation.py
6. swin_transformer.py

这些文件与DAB-DETR都是相同的


transformer.py文件中，在Transformer类的forward方法中多了两个参数，tgt和attn_mask，
这两个参数是在进入forward前生成的
其他transformer文件中的内容是相同的

---

prepare_for_dn, dn_post_process, compute_dn_loss

主要的不同在DABDETR.py文件中

在DABDETR的forward方法中多了prepare_for_dn和dn_post_process

在SetCriterion的forward方法中多了compute_dn_loss

论文中最主要的内容都是在prepare_for_dn方法中




