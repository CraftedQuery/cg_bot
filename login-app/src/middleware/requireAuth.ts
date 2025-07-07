import { Request, Response, NextFunction } from 'express';
import prisma from '../prisma';
import { verifyToken } from '../auth/jwt';

export async function requireAuth(
  req: Request,
  res: Response,
  next: NextFunction,
): Promise<void> {
  const token = req.cookies.session;
  if (!token) {
    res.status(401).json({ error: 'Unauthorized' });
    return;
  }
  const payload = verifyToken(token);
  if (!payload) {
    res.status(401).json({ error: 'Unauthorized' });
    return;
  }
  const user = await prisma.user.findUnique({ where: { id: payload.sub } });
  if (!user) {
    res.status(401).json({ error: 'Unauthorized' });
    return;
  }
  (req as any).user = user;
  next();
}
