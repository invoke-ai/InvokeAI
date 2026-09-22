const IMAGE_STORAGE_MAINTENANCE_ACTIVE_DETAIL = 'Image storage maintenance is active';

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null;

export const getImageUploadFailedDescription = (
  errorMessage: string | undefined,
  payload: unknown,
  maintenanceMessage: string
): string | undefined => {
  if (isRecord(payload) && payload.status === 409 && isRecord(payload.data)) {
    if (payload.data.detail === IMAGE_STORAGE_MAINTENANCE_ACTIVE_DETAIL) {
      return maintenanceMessage;
    }
  }
  return errorMessage;
};
