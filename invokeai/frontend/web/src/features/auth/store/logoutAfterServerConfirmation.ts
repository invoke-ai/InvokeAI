export const logoutAfterServerConfirmation = async (
  logoutOnServer: () => Promise<unknown>,
  clearLocalAuthentication: () => void
) => {
  await logoutOnServer();
  clearLocalAuthentication();
};
