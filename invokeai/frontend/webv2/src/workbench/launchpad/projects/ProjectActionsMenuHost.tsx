import type { ProjectSummary } from '@workbench/projects/library';
import type { MouseEvent, ReactNode } from 'react';

import { Menu, Portal } from '@chakra-ui/react';
import { INTERMEDIATES_SETTING_ID, requestIntermediatesFocus } from '@features/intermediates';
import { RenameDialog } from '@platform/ui/RenameDialog';
import { useNavigate } from '@tanstack/react-router';
import { isProjectSummaryCompatible } from '@workbench/projects/library';
import { createContext, lazy, Suspense, useCallback, useContext, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { ProjectCardActions } from './useProjectCardActions';

import { ProjectActionsMenuBody } from './ProjectActionsMenu';
import { useProjectCardActions } from './useProjectCardActions';

// The dialog reads the gallery's board query, and this host sits in a boot graph the performance gates pin by source
// file, so even a shared three-line wrapper module would register as growth. Loaded the first time a delete is asked for.
const LazyDeleteProjectDialog = lazy(() =>
  import('@workbench/projects/components/DeleteProjectDialog').then((module) => ({
    default: module.DeleteProjectDialog,
  }))
);

/**
 * Use one menu host so switching cards cannot race Zag's nested-layer teardown. Keep dialogs beside the menu so
 * selecting an action does not unmount them.
 */

interface ProjectMenuTarget {
  isPinned: boolean;
  summary: ProjectSummary;
  onTogglePin: (projectId: string) => void;
}

type MenuAnchor =
  | { kind: 'pointer'; x: number; y: number }
  | { kind: 'trigger'; rect: { height: number; width: number; x: number; y: number } };

interface MenuRequest extends ProjectMenuTarget {
  anchor: MenuAnchor;
  /** Where focus goes when the menu closes: without a registered zag trigger,
   * the machine's own focus restore has nothing to return to. */
  returnFocus: HTMLElement | null;
  /** Distinguishes successive opens so the hosted menu remounts even when the
   * anchor repeats — zag may have internally closed the previous machine. */
  ticket: number;
}

interface DialogRequest {
  actions: ProjectCardActions;
  kind: 'delete' | 'rename';
  name: string;
  projectId: string;
}

interface ProjectActionsMenuControl {
  /** The project whose menu is showing; drives the dots triggers' `aria-expanded`. */
  activeProjectId: string | null;
  openAtPointer: (event: MouseEvent, target: ProjectMenuTarget) => void;
  openFromTrigger: (element: HTMLElement, target: ProjectMenuTarget) => void;
}

const ProjectActionsMenuContext = createContext<ProjectActionsMenuControl | null>(null);

export const useProjectActionsMenu = (): ProjectActionsMenuControl => {
  const control = useContext(ProjectActionsMenuContext);

  if (!control) {
    throw new Error('useProjectActionsMenu must be used within a ProjectActionsMenuProvider');
  }

  return control;
};

export const ProjectActionsMenuProvider = ({ children }: { children: ReactNode }) => {
  const { t } = useTranslation();
  const [menuRequest, setMenuRequest] = useState<MenuRequest | null>(null);
  const [dialogRequest, setDialogRequest] = useState<DialogRequest | null>(null);
  // Once mounted the dialog stays, so its close animation and the lazy chunk are paid for once.
  const [hasRequestedDelete, setHasRequestedDelete] = useState(false);
  const ticketRef = useRef(0);

  const closeMenu = useCallback(() => setMenuRequest(null), []);
  const closeDialog = useCallback(() => setDialogRequest(null), []);
  const requestDialog = useCallback((dialog: DialogRequest) => {
    if (dialog.kind === 'delete') {
      setHasRequestedDelete(true);
    }
    setDialogRequest(dialog);
  }, []);
  const openAtPointer = useCallback((event: MouseEvent, target: ProjectMenuTarget) => {
    event.preventDefault();
    ticketRef.current += 1;
    setMenuRequest({
      ...target,
      anchor: { kind: 'pointer', x: event.clientX, y: event.clientY },
      returnFocus: (event.currentTarget as HTMLElement).querySelector<HTMLElement>('a, button, [tabindex]'),
      ticket: ticketRef.current,
    });
  }, []);
  const openFromTrigger = useCallback((element: HTMLElement, target: ProjectMenuTarget) => {
    const { height, width, x, y } = element.getBoundingClientRect();

    ticketRef.current += 1;
    setMenuRequest({
      ...target,
      anchor: { kind: 'trigger', rect: { height, width, x, y } },
      returnFocus: element,
      ticket: ticketRef.current,
    });
  }, []);

  const control = useMemo(
    () => ({ activeProjectId: menuRequest?.summary.id ?? null, openAtPointer, openFromTrigger }),
    [menuRequest?.summary.id, openAtPointer, openFromTrigger]
  );

  const menuKey = menuRequest ? `${menuRequest.summary.id}:${menuRequest.ticket}` : '';

  return (
    <ProjectActionsMenuContext.Provider value={control}>
      {children}
      {menuRequest ? (
        <HostedProjectActionsMenu
          key={menuKey}
          request={menuRequest}
          onClose={closeMenu}
          onRequestDialog={requestDialog}
        />
      ) : null}

      <RenameDialog
        initialName={dialogRequest?.kind === 'rename' ? dialogRequest.name : ''}
        isOpen={dialogRequest?.kind === 'rename'}
        label={t('projects.renameProjectNameLabel')}
        submitLabel={t('common.rename')}
        title={t('projects.renameProject')}
        onClose={closeDialog}
        onSubmit={dialogRequest?.actions.rename ?? NOOP_SUBMIT}
      />

      {hasRequestedDelete ? (
        <Suspense fallback={null}>
          <LazyDeleteProjectDialog
            body={t('projects.deleteProjectCardBody', { name: dialogRequest?.name ?? '' })}
            isOpen={dialogRequest?.kind === 'delete'}
            projectId={dialogRequest?.kind === 'delete' ? dialogRequest.projectId : null}
            onClose={closeDialog}
            onConfirm={dialogRequest?.actions.delete ?? NOOP_SUBMIT}
          />
        </Suspense>
      ) : null}
    </ProjectActionsMenuContext.Provider>
  );
};

const NOOP_SUBMIT = () => Promise.resolve();

/** Remember open state on pointerdown so outside dismissal followed by click toggles closed instead of reopening. */
export const useProjectActionsMenuTrigger = (target: ProjectMenuTarget) => {
  const menu = useProjectActionsMenu();
  const suppressNextOpenRef = useRef(false);
  const isExpanded = menu.activeProjectId === target.summary.id;

  const onPointerDown = useCallback(() => {
    if (menu.activeProjectId === target.summary.id) {
      suppressNextOpenRef.current = true;
    }
  }, [menu.activeProjectId, target.summary.id]);
  const onClick = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      if (suppressNextOpenRef.current) {
        suppressNextOpenRef.current = false;

        return;
      }

      menu.openFromTrigger(event.currentTarget, target);
    },
    [menu, target]
  );

  return { isExpanded, onClick, onPointerDown };
};

const HostedProjectActionsMenu = ({
  request,
  onClose,
  onRequestDialog,
}: {
  request: MenuRequest;
  onClose: () => void;
  onRequestDialog: (dialog: DialogRequest) => void;
}) => {
  const actions = useProjectCardActions(request.summary);
  const navigate = useNavigate();
  const isCompatible = isProjectSummaryCompatible(request.summary);
  const projectSearch = useMemo(() => ({ project: request.summary.id }), [request.summary.id]);
  const positioning = useMemo(() => {
    const anchor = request.anchor;

    return anchor.kind === 'pointer'
      ? {
          getAnchorRect: () => ({ height: 1, width: 1, x: anchor.x, y: anchor.y }),
          placement: 'bottom-start' as const,
        }
      : {
          getAnchorRect: () => anchor.rect,
          placement: 'bottom-end' as const,
        };
  }, [request.anchor]);

  const handleOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (!event.open) {
        // Focus would otherwise drop to the body (Escape, item select). When
        // an outside click moved it already, leave the browser's target alone.
        if (document.activeElement?.closest('[data-scope="menu"]')) {
          request.returnFocus?.focus();
        }

        onClose();
      }
    },
    [onClose, request.returnFocus]
  );
  const handleRename = useCallback(
    () => onRequestDialog({ actions, kind: 'rename', name: request.summary.name, projectId: request.summary.id }),
    [actions, onRequestDialog, request.summary.id, request.summary.name]
  );
  const handleDeleteIntermediates = useCallback(() => {
    // The manager lives in Preferences; it reads this intent when it starts.
    requestIntermediatesFocus({ projectId: request.summary.id });
    onClose();
    void navigate({
      params: { section: 'intermediates' },
      search: { setting: INTERMEDIATES_SETTING_ID },
      to: '/preferences/$section',
    });
  }, [navigate, onClose, request.summary.id]);
  const handleDelete = useCallback(
    () => onRequestDialog({ actions, kind: 'delete', name: request.summary.name, projectId: request.summary.id }),
    [actions, onRequestDialog, request.summary.id, request.summary.name]
  );
  const handleDuplicate = useCallback(() => void actions.duplicate(), [actions]);
  const handleExport = useCallback(() => void actions.export(), [actions]);
  const handleTogglePin = useCallback(() => request.onTogglePin(request.summary.id), [request]);

  return (
    <Menu.Root open positioning={positioning} onOpenChange={handleOpenChange}>
      <Portal>
        <Menu.Positioner>
          <ProjectActionsMenuBody
            isPinned={request.isPinned}
            isCompatible={isCompatible}
            projectSearch={projectSearch}
            onDeleteIntermediates={handleDeleteIntermediates}
            onDelete={handleDelete}
            onDuplicate={handleDuplicate}
            onExport={handleExport}
            onRename={handleRename}
            onTogglePin={handleTogglePin}
          />
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};
