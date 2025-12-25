pub fn iou(box1: &[f32; 4], box2: &[f32; 4]) -> f32 {
    let box1_tlbr = tlwh_to_tlbr(box1);
    let box2_tlbr = tlwh_to_tlbr(box2);

    let x1 = box1_tlbr[0].max(box2_tlbr[0]);
    let y1 = box1_tlbr[1].max(box2_tlbr[1]);
    let x2 = box1_tlbr[2].min(box2_tlbr[2]);
    let y2 = box1_tlbr[3].min(box2_tlbr[3]);

    let w = (x2 - x1).max(0.0);
    let h = (y2 - y1).max(0.0);
    let inter_area = w * h;

    let area1 = box1[2] * box1[3];
    let area2 = box2[2] * box2[3];

    let union_area = area1 + area2 - inter_area;

    if union_area <= 0.0 {
        return 0.0;
    }
    inter_area / union_area
}

pub fn tlwh_to_tlbr(tlwh: &[f32; 4]) -> [f32; 4] {
    [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]]
}

pub fn iou_batch(bboxes1: &[[f32; 4]], bboxes2: &[[f32; 4]]) -> Vec<Vec<f32>> {
    let mut iou_matrix = vec![vec![0.0; bboxes2.len()]; bboxes1.len()];
    for (i, box1) in bboxes1.iter().enumerate() {
        for (j, box2) in bboxes2.iter().enumerate() {
            iou_matrix[i][j] = iou(box1, box2);
        }
    }
    iou_matrix
}
