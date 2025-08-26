import type { Middleware } from '@reduxjs/toolkit';
import type { StudioInitAction } from 'app/hooks/useStudioInitAction';
import type { LoggingOverrides } from 'app/logging/logger';
import type { CustomStarUi } from 'app/store/nanostores/customStarUI';
import type { PartialAppConfig } from 'app/types/invokeai';
import type { SocketOptions } from 'dgram';
import type { WorkflowSortOption, WorkflowTagCategory } from 'features/nodes/store/workflowLibrarySlice';
import type { WorkflowCategory } from 'features/nodes/types/workflow';
import type { ToastConfig } from 'features/toast/toast';
import type { PropsWithChildren, ReactNode } from 'react';
import type { ManagerOptions } from 'socket.io-client';

export interface InvokeAIUIProps extends PropsWithChildren {
  apiUrl?: string;
  openAPISchemaUrl?: string;
  token?: string;
  config?: PartialAppConfig;
  customNavComponent?: ReactNode;
  accountSettingsLink?: string;
  middleware?: Middleware[];
  projectId?: string;
  projectName?: string;
  projectUrl?: string;
  queueId?: string;
  studioInitAction?: StudioInitAction;
  customStarUi?: CustomStarUi;
  socketOptions?: Partial<ManagerOptions & SocketOptions>;
  isDebugging?: boolean;
  logo?: ReactNode;
  toastMap?: Record<string, ToastConfig>;
  accountTypeText?: string;
  videoUpsellComponent?: ReactNode;
  whatsNew?: ReactNode[];
  workflowCategories?: WorkflowCategory[];
  workflowTagCategories?: WorkflowTagCategory[];
  workflowSortOptions?: WorkflowSortOption[];
  loggingOverrides?: LoggingOverrides;
  /**
   * If provided, overrides in-app navigation to the model manager
   */
  onClickGoToModelManager?: () => void;
  storagePersistDebounce?: number;
}
