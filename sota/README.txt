SOTA REQUIREMENT: NOT YET COMPLETED

The assignment names Step-Video-T2V or ArtAug for hand gestures. Neither model was executed in this project. Classical rotation, noise and brightness are not ArtAug. MobileNetV3 is the requested classifier, not the specified SOTA add-on.

No CUDA GPU is exposed in the execution environment. No authentic generated gesture clips were provided. We therefore do not fabricate model outputs, provenance or SOTA performance.

Actual continuation route:
1. Use the official Step-Video-T2V repository on suitable GPU infrastructure:
   https://github.com/stepfun-ai/Step-Video-T2V
   Follow its current installation, model weights, hardware and inference instructions.
2. Generate separate clips for peace, stop and thumbs_up, varying background and camera motion. Use anonymous, consent-safe hands. Keep exact prompt, seed, model revision and generation settings.
   Example prompt: Close-up of one adult hand making a clear peace sign, exactly two raised fingers, stationary gesture, plain wall, slow lateral camera movement, realistic lighting.
3. Manually inspect every sampled frame. Reject malformed fingers, ambiguous gestures and clips that change gesture. A text prompt is not a reliable ground-truth label.
4. Create clips.json next to the files, for example:
   [{"video":"peace_wall.mp4","label":"peace","model":"Step-Video-T2V","prompt":"YOUR ACTUAL PROMPT","seed":42,"reviewed":true}]
5. Install ffmpeg, then from project root:
   python sota/evaluate_clips.py /path/to/clips.json results/synthetic_stress
6. Include the actual watermarked clips, manifest and metrics in the app/report/video. Compare synthetic stress results separately from real-scene evaluation. If retraining, assign all frames from one clip to TRAIN ONLY and preserve the existing real validation/test sets. Run a matched with/without-synthetic ablation; do not tune on the kitchen test set.

The adapter is implemented, but its Step-Video integration has not been exercised on authentic generated clips. It is preparation, not completion of the criterion.

ArtAug reading:
https://arxiv.org/abs/2412.12888
Required Neurohive discussion page:
https://neurohive.io/en/state-of-the-art/artaug-a-novel-synthesis-understanding-framework-for-image-generation-enhancement/
