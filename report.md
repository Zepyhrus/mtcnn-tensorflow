# Test on 0616
1. Test group 1: 
  * Net: RNet + NMS-0.6;
  * threshold: [0.5, 0.6, 0.9];
  * min_face_size: 24;
  * scale_factor: 0.909;
  * nms_ratio: 0.6;
2. Test group 2:
  * Net: ONet;
  * threshold: [0.5, 0.6, 0.9]
  * min_face_size: 24;
  * scale_factor: 0.909;
3. Test group 3:
  * Net: ONet;
  * threshold: [0.5, 0.6, 0.8]
  * min_face_size: 24;
  * scale_factor: 0.909;

## Conclusion
* RNet: 11 missing;
* ONet-0.8: 117 missing;
* ONet-0.9: 170 missing;
* ONet is not trained;
* ONet: 
  * landmark: 595k;
  * pos: 54k, pos is too few;
  * neg: 273k;
  * part: 67k, part is too few;
