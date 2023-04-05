/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $PaginatedResults_ImageResponse_ = {
  description: `Paginated results`,
  properties: {
    items: {
      type: 'array',
      contains: {
        type: 'ImageResponse',
      },
      isRequired: true,
    },
    page: {
      type: 'number',
      description: `Current Page`,
      isRequired: true,
    },
    pages: {
      type: 'number',
      description: `Total number of pages`,
      isRequired: true,
    },
    per_page: {
      type: 'number',
      description: `Number of items per page`,
      isRequired: true,
    },
    total: {
      type: 'number',
      description: `Total number of items in result`,
      isRequired: true,
    },
  },
} as const;
