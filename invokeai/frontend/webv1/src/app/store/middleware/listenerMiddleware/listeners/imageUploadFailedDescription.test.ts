import { describe, expect, it } from 'vitest';

import { getImageUploadFailedDescription } from './imageUploadFailedDescription';

describe('getImageUploadFailedDescription', () => {
  it('explains that image maintenance blocked the upload', () => {
    expect(
      getImageUploadFailedDescription(
        'Rejected',
        { status: 409, data: { detail: 'Image storage maintenance is active' } },
        'Recover image storage maintenance before retrying the upload.'
      )
    ).toBe('Recover image storage maintenance before retrying the upload.');
  });

  it('preserves unrelated upload errors', () => {
    expect(getImageUploadFailedDescription('Request failed with status code 500', { status: 500 }, 'maintenance')).toBe(
      'Request failed with status code 500'
    );
  });
});
