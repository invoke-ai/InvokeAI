import {
  assertAccountScopeCurrent,
  registerAccountOwnedResource,
  type AccountScope,
} from '@platform/state/accountLifecycle';

import {
  deleteWorkbenchDatabase,
  openUnloadJournalDatabase,
  openWorkbenchDatabase,
  type UnloadJournalDatabase,
  type WorkbenchDatabase,
} from './workbenchDatabase';

export interface AccountOwnedDatabaseLease<Database> {
  database: Database;
  release(): void;
}

export type AccountOwnedWorkbenchDatabaseLease = AccountOwnedDatabaseLease<WorkbenchDatabase>;

interface AccountDatabaseGroup {
  cleared: boolean;
  readonly connections: Set<{ close(): void }>;
}

/** One group per account lifetime; clearing it closes every connection and deletes the account's databases. */
const groups = new WeakMap<AccountScope, AccountDatabaseGroup>();

interface AcquireDependencies<Database> {
  deleteDatabase?: typeof deleteWorkbenchDatabase;
  openDatabase?: (storageSuffix: string) => Promise<Database>;
}

const acquireAccountOwnedDatabase = async <Database extends { close(): void }>(
  owner: AccountScope,
  openDatabase: (storageSuffix: string) => Promise<Database>,
  deleteDatabase: typeof deleteWorkbenchDatabase
): Promise<AccountOwnedDatabaseLease<Database> | null> => {
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

  let database: Database;
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

export const acquireAccountOwnedWorkbenchDatabase = (
  owner: AccountScope,
  {
    deleteDatabase = deleteWorkbenchDatabase,
    openDatabase = openWorkbenchDatabase,
  }: AcquireDependencies<WorkbenchDatabase> = {}
): Promise<AccountOwnedWorkbenchDatabaseLease | null> =>
  acquireAccountOwnedDatabase(owner, openDatabase, deleteDatabase);

export const acquireAccountOwnedUnloadJournalDatabase = (
  owner: AccountScope,
  {
    deleteDatabase = deleteWorkbenchDatabase,
    openDatabase = openUnloadJournalDatabase,
  }: AcquireDependencies<UnloadJournalDatabase> = {}
): Promise<AccountOwnedDatabaseLease<UnloadJournalDatabase> | null> =>
  acquireAccountOwnedDatabase(owner, openDatabase, deleteDatabase);
