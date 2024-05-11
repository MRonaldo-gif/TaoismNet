# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='TaoismNet',
        pretrain_img_size=224,

        in_chans=3,
        num_classes=2,
        #128:96,256:192,512:384,1024:768
        ##########################################
        #V6
        # dims=[96, 192, 384, 768, 768],
        # strides=[2, 2, 2, 1],
        ##########################################
        ##########################################
        #V6
        # dims=[96, 96, 192, 384,384, 768,768],
        # strides=[1 ,2 , 2 ,1 , 2, 1],
        ##########################################
        ##########################################
        #V7
        # dims=[96, 96, 192, 384, 768, 768],
        # strides=[1, 2, 2, 2, 1],
        ##########################################
        ##########################################
        #V8
        # dims=[96, 96, 192, 384, 384, 768, 768],
        # strides=[1, 2, 2, 1, 2, 1],
        ##########################################
        ##########################################
        #V9
        # dims=[96,96,192,384,384,384,768,768],
        # strides=[1, 2, 2, 1, 1, 2, 1],
        ##########################################
        ##########################################
        #V10
        # dims=[96, 96, 192, 384, 384, 384, 384， 768, 768],
        # strides=[1, 2, 2, 1, 1, 1, 2, 1],
        ##########################################
        ##########################################
        #V11
        # dims=[96, 96, 192, 384, 384, 384, 384，384， 768, 768],
        # strides=[1, 2, 2, 1, 1, 1, 1, 2, 1],
        ##########################################
        ##########################################
        #V12
        # dims=[96, 96, 192, 384, 384, 384, 384，384，384， 768, 768],
        # strides=[1, 2, 2, 1, 1, 1, 1, 1, 2, 1],
        ##########################################
        ##########################################
        #V13
        dims=[96, 96, 192, 384, 384, 384, 384,384,384,384, 768, 768],
        strides=[1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1],
        ##########################################
        ##########################################
        # strides=[2, 2, 2],
        # dims=[96 * 4, 192 * 4, 384 * 4, 768 * 4, 768 * 4],
        # dims=[128 * 4, 256 * 4, 512 * 4, 1024 * 4, 1024 * 4],


        # dims=[128 * 4, 128 * 4, 256 * 4, 512 * 4, 512 * 4, 512 * 4, 512 * 4, 512 * 4, 512 * 4, 512 * 4, 1024 * 4,
        #       1024 * 4],
        # strides=[1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1],
        drop_rate=0,
        act_num=3,
        deploy=False,
        ada_pool=None,



        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 96, 192, 384 ,384 ,768,768],

        # in_channels=[128 * 4, 256 * 4, 512 * 4, 1024 * 4, 1024 * 4],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='DiceLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='DiceLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
