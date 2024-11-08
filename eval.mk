# Variables
CONFIG = configure/ablation_framestack_4.yml
MAX_TRY = 5
MAX_SAMPLE = 100
EVAL_OUTPUT = output/evaluate_model_output
PSPI_OUTPUT = output/pspi_output
PREDICTION_NAME = /media/tien/SSD-NOT-OS/pain_intermediate_data/experiment/rolling_diffusion_1/test/frames
FRAME_OUTPUT = /media/tien/SSD-NOT-OS/pain_intermediate_data/experiment/rolling_diffusion_1/test

# Targets
all: compute_metrics

evaluate_model: $(CONFIG)
    python evaluate_model.py --config $(CONFIG) --max_try $(MAX_TRY) --max_sample $(MAX_SAMPLE) --output $(EVAL_OUTPUT)

render_frames: evaluate_model
    python render_from_exp.py --input_path "$(PREDICTION_NAME).pt" --output_dir "$(FRAME_OUTPUT)" --video_render true

extract_pspi: render_frames
    python extract_pspi.py --input $(EVAL_OUTPUT) --output $(PSPI_OUTPUT)

compute_metrics: extract_pspi
    python compute_metrics.py --max_try $(MAX_TRY) --max_sample $(MAX_SAMPLE) --input $(EVAL_OUTPUT) --pspi $(PSPI_OUTPUT)

.PHONY: all evaluate_model render_frames extract_pspi compute_metrics