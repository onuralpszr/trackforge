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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tlwh_to_tlbr() {
        let tlwh = [10.0, 20.0, 30.0, 40.0];
        let tlbr = tlwh_to_tlbr(&tlwh);
        assert_eq!(tlbr, [10.0, 20.0, 40.0, 60.0]);
    }

    #[test]
    fn test_iou_overlapping() {
        // box1: 0,0, 10,10 (area 100)
        // box2: 5,5, 10,10 (area 100)
        // intersection: 5,5, 10,10 -> w=5, h=5 -> area 25
        // union: 100 + 100 - 25 = 175
        // iou: 25 / 175 = 1/7 ~= 0.142857

        let box1 = [0.0, 0.0, 10.0, 10.0];
        let box2 = [5.0, 5.0, 10.0, 10.0];
        let val = iou(&box1, &box2);
        assert!((val - 0.142857).abs() < 1e-5);
    }

    #[test]
    fn test_iou_no_overlap() {
        let box1 = [0.0, 0.0, 10.0, 10.0];
        let box2 = [20.0, 20.0, 10.0, 10.0];
        let val = iou(&box1, &box2);
        assert_eq!(val, 0.0);
    }

    #[test]
    fn test_iou_contained() {
        let box1 = [0.0, 0.0, 100.0, 100.0];
        let box2 = [25.0, 25.0, 50.0, 50.0];

        // area 2500
        // intersection is box2 (2500)
        // union is box1 (10000)
        // iou = 0.25
        let val = iou(&box1, &box2);
        assert_eq!(val, 0.25);
    }

    #[test]
    fn test_iou_batch() {
        let boxes1 = vec![[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 10.0, 10.0]];
        let boxes2 = vec![
            [0.0, 0.0, 10.0, 10.0],     // Matches box1[0] perfectly
            [5.0, 5.0, 10.0, 10.0],     // Partially matches box1[0]
            [200.0, 200.0, 10.0, 10.0], // No match
        ];

        let ious = iou_batch(&boxes1, &boxes2);
        assert_eq!(ious.len(), 2);
        assert_eq!(ious[0].len(), 3);
        assert_eq!(ious[1].len(), 3);

        assert_eq!(ious[0][0], 1.0);
        assert!((ious[0][1] - 0.142857).abs() < 1e-4);
        assert_eq!(ious[0][2], 0.0);

        assert_eq!(ious[1][0], 0.0);
        assert_eq!(ious[1][1], 0.0);
        assert_eq!(ious[1][2], 0.0);
    }
}
