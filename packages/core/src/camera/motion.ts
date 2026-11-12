/**
 * Asks for permission to read the motion sensor (`devicemotion` events), as iOS Safari requires
 * since iOS 13: call it from a user gesture (a tap handler). Resolves to `true` where no
 * permission is needed (Android, desktop) or it was granted, `false` if it was denied, the API is
 * missing or the call was not allowed (e.g. outside a gesture).
 */
export async function requestDeviceMotionPermission(): Promise<boolean> {
  if (typeof DeviceMotionEvent === 'undefined') return false;
  const request = (DeviceMotionEvent as unknown as { requestPermission?: () => Promise<string> })
    .requestPermission;
  if (typeof request !== 'function') return true;
  try {
    const granted = (await request.call(DeviceMotionEvent)) === 'granted';
    if (granted) for (const listener of [...grantedListeners]) listener();
    return granted;
  } catch {
    return false;
  }
}

const grantedListeners = new Set<() => void>();

/**
 * Calls `listener` whenever {@link requestDeviceMotionPermission} gets a grant (internal: the
 * scanner controller re-arms its `devicemotion` listener). Returns an unsubscribe function.
 */
export function onDeviceMotionGranted(listener: () => void): () => void {
  grantedListeners.add(listener);
  return () => grantedListeners.delete(listener);
}
