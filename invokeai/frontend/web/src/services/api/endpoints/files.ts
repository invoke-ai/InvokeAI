import type { FileDTO, UploadFileArg } from 'services/api/types';

import { api, buildV1Url } from '..';

/**
 * Builds an endpoint URL for the managed files router.
 * @example
 * buildFilesUrl('some-path')
 * // '/api/v1/files/some-path'
 */
const buildFilesUrl = (path: string = '') => buildV1Url(`files/${path}`);

export const filesApi = api.injectEndpoints({
  endpoints: (build) => ({
    getFileDTO: build.query<FileDTO, string>({
      query: (file_id) => ({ url: buildFilesUrl(`i/${file_id}`) }),
      providesTags: (result, error, file_id) => [{ type: 'File', id: file_id }],
    }),
    uploadFile: build.mutation<FileDTO, UploadFileArg>({
      query: ({ file }) => {
        const formData = new FormData();
        formData.append('file', file);
        return {
          url: buildFilesUrl('upload'),
          method: 'POST',
          body: formData,
        };
      },
      invalidatesTags: (result) => (result ? [{ type: 'File', id: result.file_id }] : []),
    }),
    deleteFile: build.mutation<void, string>({
      query: (file_id) => ({
        url: buildFilesUrl(`i/${file_id}`),
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, file_id) => [{ type: 'File', id: file_id }],
    }),
  }),
});

export const { useDeleteFileMutation, useGetFileDTOQuery, useUploadFileMutation } = filesApi;
