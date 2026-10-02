import {
  assertAccountScopeCurrent,
  registerAccountOwnedResource,
  type AccountScope,
} from '@platform/state/accountLifecycle';

import { deleteWorkbenchDatabase, openWorkbenchDatabase, type WorkbenchDatabase } from './workbenchDatabase';

export interface AccountOwnedWorkbenchDatabaseLease {
  database: WorkbenchDatabase;
  release(): void;
}

interface AccountDatabaseGroup {
  cleared: boolean;
  readonly connections: Set<WorkbenchDatabase>;
}

const groups = new WeakMap<AccountScope, AccountDatabaseGroup>();

export const acquireAccountOwnedWorkbenchDatabase = async (
  owner: AccountScope,
  {
    deleteDatabase = deleteWorkbenchDatabase,
    openDatabase = openWorkbenchDatabase,
  }: {
    deleteDatabase?: typeof deleteWorkbenchDatabase;
    openDatabase?: typeof openWorkbenchDatabase;
  } = {}
): Promise<AccountOwnedWorkbenchDatabaseLease | null> => {
  assertAccountScopeCurrent(owner);
  if (owner.accountId === null) {
    throw new Error('Workbench storage requires an active account.');
  }

  let group = groups.get(owner);
  if (!group) {
    group = { cleared: false, connections: new Set() };
    groups.set(owner, group);
    let unregister: () => void = () => undefined;
    unregister = registerAccountOwnedResource({
      clear: () => {
        unregister();
        unregister = () => undefined;
        group!.cleared = true;
        for (const connection of group!.connections) {
          connection.close();
        }
        group!.connections.clear();
        void deleteDatabase(owner.storageSuffix);
      },
      name: `workbench-database:${owner.epoch}`,
    });
  }

  let database: WorkbenchDatabase;
  try {
    database = await openDatabase(owner.storageSuffix);
  } catch {
    if (group.cleared) {
      assertAccountScopeCurrent(owner);
    }
    return null;
  }

  try {
    assertAccountScopeCurrent(owner);
  } catch (error) {
    database.close();
    void deleteDatabase(owner.storageSuffix);
    throw error;
  }

  group.connections.add(database);
  let isReleased = false;
  return {
    database,
    release() {
      if (isReleased) {
        return;
      }
      isReleased = true;
      group.connections.delete(database);
      database.close();
    },
  };
};
