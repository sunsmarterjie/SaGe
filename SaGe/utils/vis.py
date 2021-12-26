import cv2
import torch
import numpy as np

def drawMatches(img1, id1, img2, id2):

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    out[:rows1,:cols1] = img1
    out[:rows2,cols1:] = img2

    for img1_idx, img2_idx in zip(id1, id2):
        # x - columns
        # y - rows
        img1_idx = [x * 32 + 16 for x in img1_idx]
        img2_idx = [x * 32 + 16 for x in img2_idx]
        x1, y1 = img1_idx
        x2, y2 = img2_idx

        cv2.circle(out, (int(np.round(x1)),int(np.round(y1))), 2, (0,255,0), 1)
        cv2.circle(out, (int(np.round(x2)+cols1),int(np.round(y2))), 2, (0,255,0), 1)
        cv2.line(out, (int(np.round(x1)),int(np.round(y1))), (int(np.round(x2)+cols1),int(np.round(y2))), (0,255,255), 1)

    return out

def vis_match(img_v1, img_v2, cos_sim_o2t, cos_sim_t2o):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    mean = torch.tensor([x * 255 for x in mean]).cuda().view(1, 3, 1, 1)
    std = torch.tensor([x * 255 for x in std]).cuda().view(1, 3, 1, 1)
    img_v1 = img_v1.mul(std).add_(mean).permute(0, 2, 3, 1).cpu().numpy()[:,:,:,::-1]
    img_v2 = img_v2.mul(std).add_(mean).permute(0, 2, 3, 1).cpu().numpy()[:,:,:,::-1]
    for i, (img1, img2, sims_o2t, sims_t2o) in enumerate(zip(img_v1, img_v2, cos_sim_o2t, cos_sim_t2o)):
        id1 = []
        id2 = []
        for j, sim in enumerate(sims_o2t):
            topk = torch.topk(sim, 1)
            for sim_v, max_id in zip(topk.values, topk.indices):
                if sim_v >= 0.7:
                    id1.append([j % 7, j // 7])
                    id2.append([int(max_id) % 7, int(max_id) // 7])

        if id1:
            out_im = drawMatches(img1, id1, img2, id2)
            cv2.imwrite('imgs/%d_o2t.jpg' % i, out_im)

        id1 = []
        id2 = []
        for j, sim in enumerate(sims_t2o):
            topk = torch.topk(sim, 1)
            for sim_v, max_id in zip(topk.values, topk.indices):
                if sim_v >= 0.7:
                    id1.append([int(max_id) % 7, int(max_id) // 7])
                    id2.append([j % 7, j // 7])

        if id1:
            out_im = drawMatches(img1, id1, img2, id2)
            cv2.imwrite('imgs/%d_t2o.jpg' % i, out_im)