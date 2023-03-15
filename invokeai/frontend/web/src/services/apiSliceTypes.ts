export enum STATUS {
  idle = 'IDLE',
  busy = 'BUSY',
  error = 'ERROR',
}

export type ProgressImage = {
  width: number;
  height: number;
  dataURL: string;
};

export interface APIState {
  sessionId: string | null;
  progressImage: ProgressImage | null;
  progress: number | null;
  status: STATUS;
}
