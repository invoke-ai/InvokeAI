#!/bin/bash
blobfuse2 mount /blob_data/ --config-file /blob_configs/config_blobfuse.yaml &
invokeai --web --host 0.0.0.0 --port 9090 --outdir /blob_data/ 
