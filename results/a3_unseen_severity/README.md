# A3 unseen transformation severity

A3 trains targeted MobileNetV3-Small models for Rotate and Brightness at seeds 42, 1337, and 2026. Final inference evaluates unseen severity conditions using transform → resize → ImageNet normalization. The notebook reports metrics, deltas, and paired McNemar/FDR results.