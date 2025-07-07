import { Router } from 'express';
import passport from '../auth/passport';
import { signToken, verifyToken } from '../auth/jwt';
import prisma from '../prisma';

const router = Router();

router.post('/local', passport.authenticate('local', { session: false }), async (req, res) => {
  const user = req.user as { id: number };
  const token = signToken(user.id);
  res.cookie('session', token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
  });
  res.json({ ok: true });
});

router.post('/logout', (req, res) => {
  res.clearCookie('session');
  res.json({ ok: true });
});

router.get('/me', async (req, res) => {
  const token = req.cookies.session;
  if (!token) return res.status(401).json({ error: 'Unauthorized' });
  const payload = verifyToken(token);
  if (!payload) return res.status(401).json({ error: 'Unauthorized' });
  const user = await prisma.user.findUnique({ where: { id: payload.sub } });
  if (!user) return res.status(401).json({ error: 'Unauthorized' });
  res.json({ id: user.id, email: user.email });
});

export default router;
