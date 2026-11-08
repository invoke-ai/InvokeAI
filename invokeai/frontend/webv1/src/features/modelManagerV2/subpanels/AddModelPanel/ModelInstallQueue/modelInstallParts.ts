import type { ModelInstallJob } from 'services/api/types';

type DownloadPart = NonNullable<ModelInstallJob['download_parts']>[number];

const nonEmptyLines = (text: string | null | undefined): string[] =>
  (text ?? '')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);

/** Parts that errored or whose server refused to resume them. */
export const getProblemDownloadParts = (job: ModelInstallJob): DownloadPart[] =>
  job.download_parts?.filter((part) => part.resume_required || part.status === 'error') ?? [];

/**
 * Why an errored part failed: `error_type` names the exception and its message, `error` is a traceback whose last
 * line does the same. Resume-required parts are explained by the caller.
 */
export const describeDownloadPartError = (part: DownloadPart): string | null =>
  nonEmptyLines(part.error_type)[0] ?? nonEmptyLines(part.error).at(-1) ?? part.resume_message ?? null;

/**
 * Resume keeps partial files and re-requests the rest by range, which also recovers parts that failed transiently.
 * Only a part whose server refused the range needs a restart, which discards partial files.
 */
export const canResumeDownload = (job: ModelInstallJob): boolean =>
  !(job.download_parts ?? []).some((part) => part.resume_required);
