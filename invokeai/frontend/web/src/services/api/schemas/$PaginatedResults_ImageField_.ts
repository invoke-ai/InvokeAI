/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $PaginatedResults_ImageField_ = {
  description: `Paginated results`,
  properties: {
    items: {
      type: 'array',
      contains: {
        type: 'ImageField',
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
