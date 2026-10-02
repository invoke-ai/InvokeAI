import { useMountEffect } from '@platform/react/useMountEffect';
import { type AccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import {
  lazy,
  Suspense,
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';

interface ExportOptions {
  includeFonts: boolean;
}

interface ProjectFileOptionsControl {
  requestExportOptions: (name: string, owner: AccountScope) => Promise<ExportOptions | null>;
  requestReferencesOnlyImport: (owner: AccountScope) => Promise<boolean>;
}

export interface ProjectFileOptionsRequest {
  ticket: number;
  kind: 'export' | 'import';
  name: string;
  returnFocus: HTMLElement | null;
  settle: (result: ExportOptions | null) => void;
}

const ProjectFileOptionsDialog = lazy(() =>
  import('./ProjectFileOptionsDialog').then((module) => ({ default: module.ProjectFileOptionsDialog }))
);

const ProjectFileOptionsContext = createContext<ProjectFileOptionsControl | null>(null);

export const useProjectFileOptions = (): ProjectFileOptionsControl => {
  const control = useContext(ProjectFileOptionsContext);
  if (!control) {
    throw new Error('Project file actions require ProjectFileOptionsProvider.');
  }
  return control;
};

export const ProjectFileOptionsProvider = ({ children }: { children: ReactNode }) => {
  const [request, setRequest] = useState<ProjectFileOptionsRequest | null>(null);
  const current = useRef<ProjectFileOptionsRequest | null>(null);
  const ticket = useRef(0);
  const requestOptions = useCallback((kind: ProjectFileOptionsRequest['kind'], name: string, owner: AccountScope) => {
    current.current?.settle(null);
    if (!isAccountScopeCurrent(owner)) {
      return Promise.resolve(null);
    }
    return new Promise<ExportOptions | null>((resolve) => {
      const requestTicket = ++ticket.current;
      const abort = () => current.current?.settle(null);
      const next: ProjectFileOptionsRequest = {
        ticket: requestTicket,
        kind,
        name,
        returnFocus: document.activeElement instanceof HTMLElement ? document.activeElement : null,
        settle: (result) => {
          owner.signal.removeEventListener('abort', abort);
          if (current.current?.ticket === requestTicket) {
            current.current = null;
            setRequest(null);
          }
          resolve(isAccountScopeCurrent(owner) ? result : null);
        },
      };
      current.current = next;
      owner.signal.addEventListener('abort', abort, { once: true });
      setRequest(next);
    });
  }, []);
  useMountEffect(() => () => current.current?.settle(null));
  const control = useMemo<ProjectFileOptionsControl>(
    () => ({
      requestExportOptions: (name, owner) => requestOptions('export', name, owner),
      requestReferencesOnlyImport: async (owner) => (await requestOptions('import', '', owner)) !== null,
    }),
    [requestOptions]
  );

  return (
    <ProjectFileOptionsContext.Provider value={control}>
      {children}
      {request ? (
        <Suspense fallback={null}>
          <ProjectFileOptionsDialog key={request.ticket} request={request} />
        </Suspense>
      ) : null}
    </ProjectFileOptionsContext.Provider>
  );
};
