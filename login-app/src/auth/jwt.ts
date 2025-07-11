import jwt from 'jsonwebtoken';

export function signToken(userId: number): string {
  return jwt.sign({ sub: userId }, process.env.JWT_SECRET as string, {
    expiresIn: '1h',
  });
}

export function verifyToken(token: string): { sub: number } | null {
  try {
    return jwt.verify(token, process.env.JWT_SECRET as string) as { sub: number };
  } catch {
    return null;
  }
}
