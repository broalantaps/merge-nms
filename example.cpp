// NMS
std::vector<int> indices;
cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);

vector<Object> tmp_objects;  // save targets which have processed by nms
for(int valid_index:indices){
    if(valid_index <= objects.size()){
        tmp_objects.push_back(objects[valid_index]);
    }
}
// test_github
#ifdef merge_nms
double merge_iou_thresold = 0.6;

for(int i = 0; i < tmp_objects.size(); ++i)
{
    double weight_sum = 0; // all targets weight sum, which calculate through iou * confidence(prob)
    for(int j = 0; j < objects.size(); ++j)
    {
        double tmp_iou = cal_iou(tmp_objects[i].rect, objects[j].rect);
        if(tmp_iou > merge_iou_thresold)
        {
            weight_sum += tmp_iou * objects[j].prob;
        }
    }
    if(weight_sum > 0)
    {
        // in my project, I have eight key points need to process
        for(int k = 0; k < 8; ++k)
        {
            double locate_val = 0;
            for(int j = 0; j < objects.size(); ++j)
            {
                double tmp_iou = cal_iou(tmp_objects[i].rect, objects[j].rect);
                if(tmp_iou > merge_iou_thresold)
                {
                    locate_val += objects[j].landmarks[k] * tmp_iou * objects[j].prob / weight_sum;  // you need to add val to locate_val decided by weight
                }
            }
            tmp_objects[i].landmarks[k] = locate_val;
        }
    }
}
#endif
