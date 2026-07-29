export const logoutAfterServerConfirmation = async (
  logoutOnServer: () => Promise<unknown>,
  clearLocalAuthentication: () => void,
  pauseMediaCookieRefresh: () => Promise<() => void> = () => Promise.resolve(() => undefined)
) => {
  const resumeMediaCookieRefresh = await pauseMediaCookieRefresh();
  try {
    await logoutOnServer();
  } catch (error) {
    resumeMediaCookieRefresh();
    throw error;
  }
  clearLocalAuthentication();
};
