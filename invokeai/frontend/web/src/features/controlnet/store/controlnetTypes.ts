import { ImageDTO } from 'services/api';

export type ControlNetUIState = {
  controlnet1: ControlNetConfig;
  controlnet2: ControlNetConfig;
  controlnet3: ControlNetConfig;
};

export type ControlNetConfig = {
  controlnetEnabled: boolean;
  controlnetImage: ImageDTO | null;
  controlnetProcessor: ControlNetProcessorTypes;
  controlnetModel: string | null | undefined;
  controlnetWeight: number;
  controlnetStart: number;
  controlnetEnd: number;
};

export type ControlNetProcessorTypes =
  | 'none'
  | 'canny'
  | 'depth'
  | 'depth_zoe'
  | 'lineart'
  | 'lineart_anime'
  | 'mediapipe'
  | 'mlsd'
  | 'normal_bae'
  | 'openpose'
  | 'pidi'
  | 'softedge'
  | 'shuffle';
