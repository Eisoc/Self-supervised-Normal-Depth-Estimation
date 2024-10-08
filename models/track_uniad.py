#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from utils.utils_uniad.grid_mask import GridMask
import copy
import math
from utils.utils_uniad.util_bbox import normalize_bbox
from mmdet.models import build_loss
from einops import rearrange
from mmdet.models.utils.transformer import inverse_sigmoid
from models.track_head_plugin import MemoryBank, QueryInteractionModule, Instances, RuntimeTrackerBase
from models.register_module import nms_free_coder, detr3d_track_coder, hungarian_assigner_3d_track, track_loss, match_cost
from models.register_module.transformer_modules import custom_base_transformer_layer, decoder, encoder, multi_scale_deformable_attn_function, spatial_cross_attention, temporal_self_attention, transformer

@DETECTORS.register_module()
class UniADTrack(MVXTwoStageDetector):
    """UniAD tracking part
    """
    def __init__(
        self, 
        use_grid_mask=False,
        img_backbone=None,
        img_neck=None,
        pts_bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
        loss_cfg=None,
        qim_args=dict(
            qim_type="QIMBase",
            merger_dropout=0,
            update_query_pos=False,
            fp_ratio=0.3,
            random_drop=0.1,
        ),
        mem_args=dict(
            memory_bank_type="MemoryBank",
            memory_bank_score_thresh=0.0,
            memory_bank_len=4,
        ),
        bbox_coder=dict(
            type="DETRTrack3DCoder",
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=300,
            num_classes=10,
            score_threshold=0.0,
            with_nms=False,
            iou_thres=0.3,
        ),
        pc_range=None,
        embed_dims=256,
        num_query=900,
        num_classes=10,
        vehicle_id_list=None,
        score_thresh=0.2,
        filter_score_thresh=0.1,
        miss_tolerance=5,
        gt_iou_threshold=0.0,
        freeze_img_backbone=False,
        freeze_img_neck=False,
        freeze_bn=False,
        freeze_bev_encoder=False,
        queue_length=3,
    ):
        super(UniADTrack, self).__init__(
            img_backbone=img_backbone,
            img_neck=img_neck,
            pts_bbox_head=pts_bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
        )

        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
        )
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.num_query = num_query
        self.num_classes = num_classes
        self.vehicle_id_list = vehicle_id_list
        self.pc_range = pc_range
        self.queue_length = queue_length
        if freeze_img_backbone:
            if freeze_bn:
                self.img_backbone.eval()
            for param in self.img_backbone.parameters():
                param.requires_grad = False
        
        if freeze_img_neck:
            if freeze_bn:
                self.img_neck.eval()
            for param in self.img_neck.parameters():
                param.requires_grad = False

        # temporal
        # self.video_test_mode = video_test_mode
        # assert self.video_test_mode

        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }
        self.query_embedding = nn.Embedding(self.num_query+1, self.embed_dims * 2)   # the final one is ego query, which constantly models ego-vehicle
        self.reference_points = nn.Linear(self.embed_dims, 3)

        self.mem_bank_len = mem_args["memory_bank_len"]
        self.track_base = RuntimeTrackerBase(
            score_thresh=score_thresh,
            filter_score_thresh=filter_score_thresh,
            miss_tolerance=miss_tolerance,
        )  # hyper-param for removing inactive queries

        self.query_interact = QueryInteractionModule(
            qim_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )

        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.memory_bank = MemoryBank(
            mem_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )
        self.mem_bank_len = (
            0 if self.memory_bank is None else self.memory_bank.max_his_length
        )
        self.criterion = build_loss(loss_cfg)
        self.test_track_instances = None
        self.l2g_r_mat = None
        self.l2g_t = None
        self.gt_iou_threshold = gt_iou_threshold
        self.bev_h, self.bev_w = self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w
        self.freeze_bev_encoder = freeze_bev_encoder

    def extract_img_feat(self, img, len_queue=None):
        """Extract features of images."""
        img = img.unsqueeze(1) # BCHW->BNCHW, N:num_cam=1
        if img is None:
            return None
        assert img.dim() == 5
        B, N, C, H, W = img.size()
        img = img.reshape(B * N, C, H, W)
        if self.use_grid_mask:
            img = self.grid_mask(img)
        self.img_backbone = self.img_backbone.to('cuda:1')
        self.img_neck = self.img_neck.to('cuda:1')
        img_feats = self.img_backbone(img.float())
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            _, c, h, w = img_feat.size()
            if len_queue is not None:
                img_feat_reshaped = img_feat.view(B//len_queue, len_queue, N, c, h, w)
            else:
                img_feat_reshaped = img_feat.view(B, N, c, h, w)
            img_feats_reshaped.append(img_feat_reshaped)
        return img_feats_reshaped

    def _generate_empty_tracks(self):
        track_instances = Instances((1, 1))
        num_queries, dim = self.query_embedding.weight.shape  # (300, 256 * 2)
        device = self.query_embedding.weight.device
        query = self.query_embedding.weight
        track_instances.ref_pts = self.reference_points(query[..., : dim // 2])

        # init boxes: xy, wl, z, h, sin, cos, vx, vy, vz
        pred_boxes_init = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device
        )
        track_instances.query = query

        track_instances.output_embedding = torch.zeros(
            (num_queries, dim >> 1), device=device
        )

        track_instances.obj_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device
        )
        track_instances.matched_gt_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device
        )
        track_instances.disappear_time = torch.zeros(
            (len(track_instances),), dtype=torch.long, device=device
        )

        track_instances.iou = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        # xy, wl, z, h, sin, cos, vx, vy, vz
        track_instances.pred_boxes = pred_boxes_init

        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device
        )

        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros(
            (len(track_instances), mem_bank_len, dim // 2),
            dtype=torch.float32,
            device=device,
        )
        track_instances.mem_padding_mask = torch.ones(
            (len(track_instances), mem_bank_len), dtype=torch.bool, device=device
        )
        track_instances.save_period = torch.zeros(
            (len(track_instances),), dtype=torch.float32, device=device
        )

        return track_instances.to(self.query_embedding.weight.device)

    def velo_update(
        self, ref_pts, velocity, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta
    ):
        """
        Args:
            ref_pts (Tensor): (num_query, 3).  in inevrse sigmoid space
            velocity (Tensor): (num_query, 2). m/s
                in lidar frame. vx, vy
            global2lidar (np.Array) [4,4].
        Outs:
            ref_pts (Tensor): (num_query, 3).  in inevrse sigmoid space
        """
        # print(l2g_r1.type(), l2g_t1.type(), ref_pts.type())
        time_delta = time_delta.type(torch.float)
        num_query = ref_pts.size(0)
        velo_pad_ = velocity.new_zeros((num_query, 1))
        velo_pad = torch.cat((velocity, velo_pad_), dim=-1)

        reference_points = ref_pts.sigmoid().clone()
        pc_range = self.pc_range
        reference_points[..., 0:1] = (
            reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        )
        reference_points[..., 1:2] = (
            reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        )
        reference_points[..., 2:3] = (
            reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        )

        reference_points = reference_points + velo_pad * time_delta

        ref_pts = reference_points @ l2g_r1 + l2g_t1 - l2g_t2

        g2l_r = torch.linalg.inv(l2g_r2).type(torch.float)

        ref_pts = ref_pts @ g2l_r

        ref_pts[..., 0:1] = (ref_pts[..., 0:1] - pc_range[0]) / (
            pc_range[3] - pc_range[0]
        )
        ref_pts[..., 1:2] = (ref_pts[..., 1:2] - pc_range[1]) / (
            pc_range[4] - pc_range[1]
        )
        ref_pts[..., 2:3] = (ref_pts[..., 2:3] - pc_range[2]) / (
            pc_range[5] - pc_range[2]
        )

        ref_pts = inverse_sigmoid(ref_pts)

        return ref_pts

    def _copy_tracks_for_loss(self, tgt_instances):
        device = self.query_embedding.weight.device
        track_instances = Instances((1, 1))

        track_instances.obj_idxes = copy.deepcopy(tgt_instances.obj_idxes)

        track_instances.matched_gt_idxes = copy.deepcopy(tgt_instances.matched_gt_idxes)
        track_instances.disappear_time = copy.deepcopy(tgt_instances.disappear_time)

        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.pred_boxes = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device
        )
        track_instances.iou = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device
        )

        track_instances.save_period = copy.deepcopy(tgt_instances.save_period)
        return track_instances.to(device)

    def get_history_bev(self, imgs_queue, img_metas_list):
        """
        Get history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()
        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_img_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev, _ = self.pts_bbox_head.get_bev_features(
                    mlvl_feats=img_feats, 
                    img_metas=img_metas, 
                    prev_bev=prev_bev)
        self.train()
        return prev_bev

    # Generate bev using bev_encoder in BEVFormer
    def get_bevs(self, imgs, img_metas, prev_img=None, prev_img_metas=None, prev_bev=None):
        if prev_img is not None and prev_img_metas is not None:
            assert prev_bev is None
            prev_bev = self.get_history_bev(prev_img, prev_img_metas)

        img_feats = self.extract_img_feat(img=imgs)
        if self.freeze_bev_encoder:
            with torch.no_grad():
                bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
                    mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev)
        else:
            bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
                    mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev)
        
        if bev_embed.shape[1] == self.bev_h * self.bev_w:
            bev_embed = bev_embed.permute(1, 0, 2)
        
        assert bev_embed.shape[0] == self.bev_h * self.bev_w
        return bev_embed, bev_pos

    @auto_fp16(apply_to=("img", "prev_bev"))
    def _forward_single_frame_train(
        self,
        img,
        img_metas,
        track_instances,
        prev_img,
        prev_img_metas,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
        all_query_embeddings=None,
        all_matched_indices=None,
        all_instances_pred_logits=None,
        all_instances_pred_boxes=None,
    ):
        """
        Perform forward only on one frame. Called in  forward_train
        Warnning: Only Support BS=1
        Args:
            img: shape [B, num_cam, 3, H, W]
            if l2g_r2 is None or l2g_t2 is None:
                it means this frame is the end of the training clip,
                so no need to call velocity update
        """
        # NOTE: You can replace BEVFormer with other BEV encoder and provide bev_embed here
        bev_embed, bev_pos = self.get_bevs(
            img, img_metas,
            prev_img=prev_img, prev_img_metas=prev_img_metas,
        )

        det_output = self.pts_bbox_head.get_detections(
            bev_embed,
            object_query_embeds=track_instances.query,
            ref_points=track_instances.ref_pts,
            img_metas=img_metas,
        )

        output_classes = det_output["all_cls_scores"]
        output_coords = det_output["all_bbox_preds"]
        output_past_trajs = det_output["all_past_traj_preds"]
        last_ref_pts = det_output["last_ref_points"]
        query_feats = det_output["query_feats"]

        out = {
            "pred_logits": output_classes[-1],
            "pred_boxes": output_coords[-1],
            "pred_past_trajs": output_past_trajs[-1],
            "ref_pts": last_ref_pts,
            "bev_embed": bev_embed,
            "bev_pos": bev_pos
        }
        with torch.no_grad():
            track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values

        # Step-1 Update track instances with current prediction
        # [nb_dec, bs, num_query, xxx]
        nb_dec = output_classes.size(0)

        # the track id will be assigned by the matcher.
        track_instances_list = [
            self._copy_tracks_for_loss(track_instances) for i in range(nb_dec - 1)
        ]
        track_instances.output_embedding = query_feats[-1][0]  # [300, feat_dim]
        velo = output_coords[-1, 0, :, -2:]  # [num_query, 3]
        if l2g_r2 is not None:
            # Update ref_pts for next frame considering each agent's velocity
            ref_pts = self.velo_update(
                last_ref_pts[0],
                velo,
                l2g_r1,
                l2g_t1,
                l2g_r2,
                l2g_t2,
                time_delta=time_delta,
            )
        else:
            ref_pts = last_ref_pts[0]

        dim = track_instances.query.shape[-1]
        track_instances.ref_pts = self.reference_points(track_instances.query[..., :dim//2])
        track_instances.ref_pts[...,:2] = ref_pts[...,:2]

        track_instances_list.append(track_instances)
        
        for i in range(nb_dec):
            track_instances = track_instances_list[i]

            track_instances.scores = track_scores
            track_instances.pred_logits = output_classes[i, 0]  # [300, num_cls]
            track_instances.pred_boxes = output_coords[i, 0]  # [300, box_dim]
            track_instances.pred_past_trajs = output_past_trajs[i, 0]  # [300,past_steps, 2]

            out["track_instances"] = track_instances
            track_instances, matched_indices = self.criterion.match_for_single_frame(
                out, i, if_step=(i == (nb_dec - 1))
            )
            all_query_embeddings.append(query_feats[i][0])
            all_matched_indices.append(matched_indices)
            all_instances_pred_logits.append(output_classes[i, 0])
            all_instances_pred_boxes.append(output_coords[i, 0])   # Not used
        
        active_index = (track_instances.obj_idxes>=0) & (track_instances.iou >= self.gt_iou_threshold) & (track_instances.matched_gt_idxes >=0)
        out.update(self.select_active_track_query(track_instances, active_index, img_metas))
        out.update(self.select_sdc_track_query(track_instances[900], img_metas))
        
        # memory bank 
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
        # Step-2 Update track instances using matcher

        tmp = {}
        tmp["init_track_instances"] = self._generate_empty_tracks()
        tmp["track_instances"] = track_instances
        out_track_instances = self.query_interact(tmp)
        out["track_instances"] = out_track_instances
        return out

    def select_active_track_query(self, track_instances, active_index, img_metas, with_mask=True):
        result_dict = self._track_instances2results(track_instances[active_index], img_metas, with_mask=with_mask)
        result_dict["track_query_embeddings"] = track_instances.output_embedding[active_index][result_dict['bbox_index']][result_dict['mask']]
        result_dict["track_query_matched_idxes"] = track_instances.matched_gt_idxes[active_index][result_dict['bbox_index']][result_dict['mask']]
        return result_dict
    
    def select_sdc_track_query(self, sdc_instance, img_metas):
        out = dict()
        result_dict = self._track_instances2results(sdc_instance, img_metas, with_mask=False)
        out["sdc_boxes_3d"] = result_dict['boxes_3d']
        out["sdc_scores_3d"] = result_dict['scores_3d']
        out["sdc_track_scores"] = result_dict['track_scores']
        out["sdc_track_bbox_results"] = result_dict['track_bbox_results']
        out["sdc_embedding"] = sdc_instance.output_embedding[0]
        return out

    @auto_fp16(apply_to=("img", "points"))
    def forward_track_train(self,
                            img,
                            gt_bboxes_3d,
                            gt_labels_3d,
                            gt_past_traj,
                            gt_past_traj_mask,
                            gt_inds,
                            gt_sdc_bbox,
                            gt_sdc_label,
                            l2g_t,
                            l2g_r_mat,
                            img_metas,
                            timestamp):
        """Forward funciton
        Args:
        Returns:
        """
        track_instances = self._generate_empty_tracks()
        num_frame = img.size(1)
        # init gt instances!
        gt_instances_list = []

        for i in range(num_frame):
            gt_instances = Instances((1, 1))
            boxes = gt_bboxes_3d[0][i].tensor.to(img.device)
            # normalize gt bboxes here!
            boxes = normalize_bbox(boxes, self.pc_range)
            sd_boxes = gt_sdc_bbox[0][i].tensor.to(img.device)
            sd_boxes = normalize_bbox(sd_boxes, self.pc_range)
            gt_instances.boxes = boxes
            gt_instances.labels = gt_labels_3d[0][i]
            gt_instances.obj_ids = gt_inds[0][i]
            gt_instances.past_traj = gt_past_traj[0][i].float()
            gt_instances.past_traj_mask = gt_past_traj_mask[0][i].float()
            gt_instances.sdc_boxes = torch.cat([sd_boxes for _ in range(boxes.shape[0])], dim=0)  # boxes.shape[0] sometimes 0
            gt_instances.sdc_labels = torch.cat([gt_sdc_label[0][i] for _ in range(gt_labels_3d[0][i].shape[0])], dim=0)
            gt_instances_list.append(gt_instances)

        self.criterion.initialize_for_single_clip(gt_instances_list)

        out = dict()

        for i in range(num_frame):
            prev_img = img[:, :i, ...] if i != 0 else img[:, :1, ...]
            prev_img_metas = copy.deepcopy(img_metas)
            # TODO: Generate prev_bev in an RNN way.

            img_single = torch.stack([img_[i] for img_ in img], dim=0)
            img_metas_single = [copy.deepcopy(img_metas[0][i])]
            if i == num_frame - 1:
                l2g_r2 = None
                l2g_t2 = None
                time_delta = None
            else:
                l2g_r2 = l2g_r_mat[0][i + 1]
                l2g_t2 = l2g_t[0][i + 1]
                time_delta = timestamp[0][i + 1] - timestamp[0][i]
            all_query_embeddings = []
            all_matched_idxes = []
            all_instances_pred_logits = []
            all_instances_pred_boxes = []
            frame_res = self._forward_single_frame_train(
                img_single,
                img_metas_single,
                track_instances,
                prev_img,
                prev_img_metas,
                l2g_r_mat[0][i],
                l2g_t[0][i],
                l2g_r2,
                l2g_t2,
                time_delta,
                all_query_embeddings,
                all_matched_idxes,
                all_instances_pred_logits,
                all_instances_pred_boxes,
            )
            # all_query_embeddings: len=dec nums, N*256
            # all_matched_idxes: len=dec nums, N*2
            track_instances = frame_res["track_instances"]
        
        get_keys = ["bev_embed", "bev_pos",
                    "track_query_embeddings", "track_query_matched_idxes", "track_bbox_results",
                    "sdc_boxes_3d", "sdc_scores_3d", "sdc_track_scores", "sdc_track_bbox_results", "sdc_embedding"]
        out.update({k: frame_res[k] for k in get_keys})
        
        losses = self.criterion.losses_dict
        return losses, out

    def upsample_bev_if_tiny(self, outs_track):
        if outs_track["bev_embed"].size(0) == 100 * 100:
            # For tiny model
            # bev_emb
            bev_embed = outs_track["bev_embed"] # [10000, 1, 256]
            dim, _, _ = bev_embed.size()
            w = h = int(math.sqrt(dim))
            assert h == w == 100

            bev_embed = rearrange(bev_embed, '(h w) b c -> b c h w', h=h, w=w)  # [1, 256, 100, 100]
            bev_embed = nn.Upsample(scale_factor=2)(bev_embed)  # [1, 256, 200, 200]
            bev_embed = rearrange(bev_embed, 'b c h w -> (h w) b c')
            outs_track["bev_embed"] = bev_embed

            # prev_bev
            prev_bev = outs_track.get("prev_bev", None)
            if prev_bev is not None:
                if self.training:
                    #  [1, 10000, 256]
                    prev_bev = rearrange(prev_bev, 'b (h w) c -> b c h w', h=h, w=w)
                    prev_bev = nn.Upsample(scale_factor=2)(prev_bev)  # [1, 256, 200, 200]
                    prev_bev = rearrange(prev_bev, 'b c h w -> b (h w) c')
                    outs_track["prev_bev"] = prev_bev
                else:
                    #  [10000, 1, 256]
                    prev_bev = rearrange(prev_bev, '(h w) b c -> b c h w', h=h, w=w)
                    prev_bev = nn.Upsample(scale_factor=2)(prev_bev)  # [1, 256, 200, 200]
                    prev_bev = rearrange(prev_bev, 'b c h w -> (h w) b c')
                    outs_track["prev_bev"] = prev_bev

            # bev_pos
            bev_pos  = outs_track["bev_pos"]  # [1, 256, 100, 100]
            bev_pos = nn.Upsample(scale_factor=2)(bev_pos)  # [1, 256, 200, 200]
            outs_track["bev_pos"] = bev_pos
        return outs_track


    def _forward_single_frame_inference(
        self,
        img,
        img_metas,
        track_instances,
        prev_bev=None,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
    ):
        """
        img: B, num_cam, C, H, W = img.shape
        """

        """ velo update """
        active_inst = track_instances[track_instances.obj_idxes >= 0]
        other_inst = track_instances[track_instances.obj_idxes < 0]

        if l2g_r2 is not None and len(active_inst) > 0 and l2g_r1 is not None:
            ref_pts = active_inst.ref_pts
            velo = active_inst.pred_boxes[:, -2:]
            ref_pts = self.velo_update(
                ref_pts, velo, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta=time_delta
            )
            ref_pts = ref_pts.squeeze(0)
            dim = active_inst.query.shape[-1]
            active_inst.ref_pts = self.reference_points(active_inst.query[..., :dim//2])
            active_inst.ref_pts[...,:2] = ref_pts[...,:2]

        track_instances = Instances.cat([other_inst, active_inst])

        # NOTE: You can replace BEVFormer with other BEV encoder and provide bev_embed here
        bev_embed, bev_pos = self.get_bevs(img, img_metas, prev_bev=prev_bev)            # 当前帧的img和上一帧率的bev，估计当前的bev编码以及pos embedding
        det_output = self.pts_bbox_head.get_detections(            # pts_bbox_head即 track head，调用self.transformer.get_states_and_refs，------------------------得到位姿！！！！！！！！！！！！！！
            bev_embed, 
            object_query_embeds=track_instances.query,
            ref_points=track_instances.ref_pts,
            img_metas=img_metas,
        )
        #         get_detections输出 {
        #     'all_cls_scores': outputs_classes,
        #     'all_bbox_preds': outputs_coords,
        #     'all_past_traj_preds': outputs_trajs,
        #     'enc_cls_scores': None,
        #     'enc_bbox_preds': None,
        #     'last_ref_points': last_ref_points,
        #     'query_feats': hs,
        # }
        output_classes = det_output["all_cls_scores"]
        output_coords = det_output["all_bbox_preds"] #-------------------------------------------"all_bbox_preds"
        last_ref_pts = det_output["last_ref_points"]
        query_feats = det_output["query_feats"]

        out = {
            "pred_logits": output_classes,
            "pred_boxes": output_coords,
            "ref_pts": last_ref_pts,
            "bev_embed": bev_embed,
            "query_embeddings": query_feats,
            "all_past_traj_preds": det_output["all_past_traj_preds"],
            "bev_pos": bev_pos,
        }

        """ update track instances with predict results """
        track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values
        # each track will be assigned an unique global id by the track base.
        track_instances.scores = track_scores
        # track_instances.track_scores = track_scores  # [300]
        track_instances.pred_logits = output_classes[-1, 0]  # [300, num_cls]
        track_instances.pred_boxes = output_coords[-1, 0]  # [300, box_dim]
        #--------------------------------------------------------------------------------------------pred_boxes！！！！！！！需要的量，包含位姿
        track_instances.output_embedding = query_feats[-1][0]  # [300, feat_dim]
        track_instances.ref_pts = last_ref_pts[0]
        # hard_code: assume the 901 query is sdc query 
        track_instances.obj_idxes[900] = -2
        """ update track base """
        self.track_base.update(track_instances, None)
       
        active_index = (track_instances.obj_idxes>=0) & (track_instances.scores >= self.track_base.filter_score_thresh)    # filter out sleep objects
        # -------------------------------这里开始使用位姿-------------------------------------------
        out.update(self.select_active_track_query(track_instances, active_index, img_metas))
        out.update(self.select_sdc_track_query(track_instances[track_instances.obj_idxes==-2], img_metas))

        """ update with memory_bank """
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)

        """  Update track instances using matcher """
        tmp = {}
        tmp["init_track_instances"] = self._generate_empty_tracks()
        tmp["track_instances"] = track_instances
        out_track_instances = self.query_interact(tmp)
        out["track_instances_fordet"] = track_instances
        #----------------------------------------------------------------------------------------------
        out["track_instances"] = out_track_instances
        out["track_obj_idxes"] = track_instances.obj_idxes
        return out

    def simple_test_track(
        self,
        img=None,
        l2g_t=None,
        l2g_r_mat=None,
        img_metas=None,
        timestamp=None,
    ):
        """only support bs=1 and sequential input"""

        bs = img.size(0)
        # img_metas = img_metas[0]

        """ init track instances for first frame """
        if (
            self.test_track_instances is None
            or img_metas[0]["scene_token"] != self.scene_token
        ):
            self.timestamp = timestamp
            # self.scene_token = img_metas[0]["scene_token"]
            self.prev_bev = None
            track_instances = self._generate_empty_tracks()
            time_delta, l2g_r1, l2g_t1, l2g_r2, l2g_t2 = None, None, None, None, None
            
        else:
            track_instances = self.test_track_instances
            time_delta = timestamp - self.timestamp
            l2g_r1 = self.l2g_r_mat
            l2g_t1 = self.l2g_t
            l2g_r2 = l2g_r_mat
            l2g_t2 = l2g_t
        
        """ get time_delta and l2g r/t infos """
        """ update frame info for next frame"""
        self.timestamp = timestamp
        self.l2g_t = l2g_t
        self.l2g_r_mat = l2g_r_mat

        """ predict and update """
        prev_bev = self.prev_bev
        frame_res = self._forward_single_frame_inference(
            img,
            img_metas,
            track_instances,
            prev_bev,
            l2g_r1,
            l2g_t1,
            l2g_r2,
            l2g_t2,
            time_delta,
        )

        self.prev_bev = frame_res["bev_embed"]
        track_instances = frame_res["track_instances"]
        track_instances_fordet = frame_res["track_instances_fordet"]
        # ----------------------------------------------------------------out["track_instances_fordet"] = track_instances, instances.pred_boxes重要，就是那个包含了位姿的量

        self.test_track_instances = track_instances
        results = [dict()]
        get_keys = ["bev_embed", "bev_pos", 
                    "track_query_embeddings", "track_bbox_results", 
                    "boxes_3d", "scores_3d", "labels_3d", "track_scores", "track_ids"]
        if self.with_motion_head:
            get_keys += ["sdc_boxes_3d", "sdc_scores_3d", "sdc_track_scores", "sdc_track_bbox_results", "sdc_embedding"]
        results[0].update({k: frame_res[k] for k in get_keys})
        results = self._det_instances2results(track_instances_fordet, results, img_metas)
        return results
    
    def _track_instances2results(self, track_instances, img_metas, with_mask=True):
        # 构建边界框字典，包含分类得分、边界框预测、跟踪分数和对象索引
        bbox_dict = dict(
            cls_scores=track_instances.pred_logits,
            bbox_preds=track_instances.pred_boxes,
            track_scores=track_instances.scores,
            obj_idxes=track_instances.obj_idxes,
        )
        
        bboxes_dict = self.bbox_coder.decode(bbox_dict, with_mask=with_mask, img_metas=img_metas)[0]
        # decoder
        # 'bboxes':boxes3d: [num_query(900),9],:cx, cy, cz, w, l, h, rot, vx, vy，
        bboxes = bboxes_dict["bboxes"]
        # 将解码后的 3D 边界框转换为特定的 3D 边界框类型
        bboxes = img_metas[0]["box_type_3d"](bboxes, 9)
        # 提取标签、分数和边界框索引
        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]
        bbox_index = bboxes_dict["bbox_index"]

        # 提取跟踪分数和对象索引
        track_scores = bboxes_dict["track_scores"]
        obj_idxes = bboxes_dict["obj_idxes"]
        # 构建结果字典，包含 3D 边界框、分数、标签、跟踪分数、边界框索引和对象索引
        result_dict = dict(
            boxes_3d=bboxes.to("cpu"),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            track_scores=track_scores.cpu(),
            bbox_index=bbox_index.cpu(),
            track_ids=obj_idxes.cpu(),
            mask=bboxes_dict["mask"].cpu(),
            track_bbox_results=[[bboxes.to("cpu"), scores.cpu(), labels.cpu(), bbox_index.cpu(), bboxes_dict["mask"].cpu()]]
        )
        # 返回结果字典
        return result_dict

    def _det_instances2results(self, instances, results, img_metas):
        """
        Outs:
        active_instances. keys:
        - 'pred_logits':
        - 'pred_boxes': normalized bboxes
        - 'scores'
        - 'obj_idxes'
        out_dict. keys:
            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
            - track_ids
            - tracking_score
        """
        # filter out sleep querys
        if instances.pred_logits.numel() == 0:
            return [None]
        bbox_dict = dict(
            cls_scores=instances.pred_logits,
            bbox_preds=instances.pred_boxes,
            track_scores=instances.scores,
            obj_idxes=instances.obj_idxes,
        )
        bboxes_dict = self.bbox_coder.decode(bbox_dict, img_metas=img_metas)[0]
        bboxes = bboxes_dict["bboxes"]
        bboxes = img_metas[0]["box_type_3d"](bboxes, 9)
        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]

        track_scores = bboxes_dict["track_scores"]
        obj_idxes = bboxes_dict["obj_idxes"]
        result_dict = results[0]
        result_dict_det = dict(
            boxes_3d_det=bboxes.to("cpu"),
            scores_3d_det=scores.cpu(),
            labels_3d_det=labels.cpu(),
        )
        if result_dict is not None:
            result_dict.update(result_dict_det)
        else:
            result_dict = None

        return [result_dict]
    
    
def get_model_cfg():
    # Unfreeze neck and BN, the from-scratch results of stage1 could be reproduced
    plugin = True
    plugin_dir = "projects/mmdet3d_plugin/"
    # If point cloud range is changed, the models should also change their point
    # cloud range accordingly
    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    voxel_size = [0.2, 0.2, 8]
    patch_size = [102.4, 102.4]
    img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
    # For nuScenes we usually do 10-class detection
    class_names = [
        "car",
        "truck",
        "construction_vehicle",
        "bus",
        "trailer",
        "barrier",
        "motorcycle",
        "bicycle",
        "pedestrian",
        "traffic_cone",
    ]

    input_modality = dict(
        use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True
    )
    _dim_ = 256
    _pos_dim_ = _dim_ // 2
    _ffn_dim_ = _dim_ * 2
    _num_levels_ = 4
    bev_h_ = 200
    bev_w_ = 200
    _feed_dim_ = _ffn_dim_
    _dim_half_ = _pos_dim_
    canvas_size = (bev_h_, bev_w_)

    # NOTE: You can change queue_length from 5 to 3 to save GPU memory, but at risk of performance drop.
    queue_length = 5  # each sequence contains `queue_length` frames.

    ### traj prediction args ###
    predict_steps = 12
    predict_modes = 6
    fut_steps = 4
    past_steps = 4
    use_nonlinear_optimizer = True

    ## occflow setting	
    occ_n_future = 4	
    occ_n_future_plan = 6
    occ_n_future_max = max([occ_n_future, occ_n_future_plan])	

    ### planning ###
    planning_steps = 6
    use_col_optim = True

    ### Occ args ### 
    occflow_grid_conf = {
        'xbound': [-50.0, 50.0, 0.5],
        'ybound': [-50.0, 50.0, 0.5],
        'zbound': [-10.0, 10.0, 20.0],
    }

    # Other settings
    train_gt_iou_threshold=0.3
    import models.track_head
    
    model = dict(
        type="UniADTrack",
        gt_iou_threshold=train_gt_iou_threshold,
        queue_length=queue_length,
        use_grid_mask=True,
        video_test_mode=True,
        num_query=900,
        num_classes=10,
        pc_range=point_cloud_range,
        img_backbone=dict(
            type="ResNet",
            depth=101,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=4,
            norm_cfg=dict(type="BN2d", requires_grad=False),
            norm_eval=True,
            style="caffe",
            dcn=dict(
                type="DCNv2", deform_groups=1, fallback_on_stride=False
            ),  # original DCNv2 will print log when perform load_state_dict
            stage_with_dcn=(False, False, True, True),
        ),
        img_neck=dict(
            type="FPN",
            in_channels=[512, 1024, 2048],
            out_channels=_dim_,
            start_level=0,
            add_extra_convs="on_output",
            num_outs=4,
            relu_before_extra_convs=True,
        ),
        freeze_img_backbone=True,
        freeze_img_neck=False,
        freeze_bn=False,
        score_thresh=0.4,
        filter_score_thresh=0.35,
        qim_args=dict(
            qim_type="QIMBase",
            merger_dropout=0,
            update_query_pos=True,
            fp_ratio=0.3,
            random_drop=0.1,
        ),  # hyper-param for query dropping mentioned in MOTR
        mem_args=dict(
            memory_bank_type="MemoryBank",
            memory_bank_score_thresh=0.0,
            memory_bank_len=4,
        ),
        loss_cfg=dict(
            type="ClipMatcher",
            num_classes=10,
            weight_dict=None,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            assigner=dict(
                type="HungarianAssigner3DTrack",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                pc_range=point_cloud_range,
            ),
            loss_cls=dict(
                type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
            ),
            loss_bbox=dict(type="L1Loss", loss_weight=0.25),
            loss_past_traj_weight=0.0,
        ),  # loss cfg for tracking

        pts_bbox_head=dict(
            type="BEVFormerTrackHead",
            bev_h=bev_h_,
            bev_w=bev_w_,
            num_query=900,
            num_classes=10,
            in_channels=_dim_,
            sync_cls_avg_factor=True,
            with_box_refine=True,
            as_two_stage=False,
            past_steps=past_steps,
            fut_steps=fut_steps,
            transformer=dict(
                type="PerceptionTransformer",
                rotate_prev_bev=True,
                use_shift=True,
                use_can_bus=True,
                embed_dims=_dim_,
                encoder=dict(
                    type="BEVFormerEncoder",
                    num_layers=6,
                    pc_range=point_cloud_range,
                    num_points_in_pillar=4,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type="BEVFormerLayer",
                        attn_cfgs=[
                            dict(
                                type="TemporalSelfAttention", embed_dims=_dim_, num_levels=1
                            ),
                            dict(
                                type="SpatialCrossAttention",
                                pc_range=point_cloud_range,
                                deformable_attention=dict(
                                    type="MSDeformableAttention3D",
                                    embed_dims=_dim_,
                                    num_points=8,
                                    num_levels=_num_levels_,
                                ),
                                embed_dims=_dim_,
                            ),
                        ],
                        feedforward_channels=_ffn_dim_,
                        ffn_dropout=0.1,
                        operation_order=(
                            "self_attn",
                            "norm",
                            "cross_attn",
                            "norm",
                            "ffn",
                            "norm",
                        ),
                    ),
                ),
                decoder=dict(
                    type="DetectionTransformerDecoder",
                    num_layers=6,
                    return_intermediate=True,
                    transformerlayers=dict(
                        type="DetrTransformerDecoderLayer",
                        attn_cfgs=[
                            dict(
                                type="MultiheadAttention",
                                embed_dims=_dim_,
                                num_heads=8,
                                dropout=0.1,
                            ),
                            dict(
                                type="CustomMSDeformableAttention",
                                embed_dims=_dim_,
                                num_levels=1,
                            ),
                        ],
                        feedforward_channels=_ffn_dim_,
                        ffn_dropout=0.1,
                        operation_order=(
                            "self_attn",
                            "norm",
                            "cross_attn",
                            "norm",
                            "ffn",
                            "norm",
                        ),
                    ),
                ),
            ),
            bbox_coder=dict(
                type="NMSFreeCoder",
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                pc_range=point_cloud_range,
                max_num=300,
                voxel_size=voxel_size,
                num_classes=10,
            ),
            positional_encoding=dict(
                type="LearnedPositionalEncoding",
                num_feats=_pos_dim_,
                row_num_embed=bev_h_,
                col_num_embed=bev_w_,
            ),
            loss_cls=dict(
                type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
            ),
            loss_bbox=dict(type="L1Loss", loss_weight=0.25),
            loss_iou=dict(type="GIoULoss", loss_weight=0.0),
        ),
        
    )
    from mmcv import Config

    cfg = Config(dict(model=model))
    return cfg