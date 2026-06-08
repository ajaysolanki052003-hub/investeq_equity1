"""Final probe — using the CORRECT Groww options symbol format from their
instruments CSV: NSE-NIFTY-DDMmmYY-strike-CE/PE
(leading-zero day, capitalized 3-letter month, 2-digit year)"""

import os, sys, time, requests
sys.path.insert(0, '/home/ajay/investeq_ajs/ema_scanner')
from groww_client import get_access_token

tok = get_access_token(os.environ['GROWW_TOTP_JWT'], os.environ['GROWW_TOTP_SECRET'])
H = {'Authorization': f'Bearer {tok}', 'Accept': 'application/json', 'X-API-VERSION': '1.0'}
URL = 'https://api.groww.in/v1/historical/candles'

def hit(sym, s, e, interval='15minute'):
    try:
        r = requests.get(URL, headers=H, timeout=15, params={
            'exchange':'NSE','segment':'FNO','groww_symbol':sym,
            'start_time':s,'end_time':e,'candle_interval':interval})
        if r.status_code == 200:
            c = r.json().get('payload', {}).get('candles') or []
            return f'200  candles={len(c)}' + (f'  first={c[0]}' if c else '')
        return f'{r.status_code}  {r.text[:120]}'
    except Exception as ex:
        return f'EXC {ex}'

# Test 1: EXPIRED contract — NIFTY 24200 CE expiring 2026-04-28 (we have its
# data from the old collector on 2026-04-23; spot was 24,184; this was THE ATM)
print('=== EXPIRED: NSE-NIFTY-28Apr26-24200-CE on 2026-04-23 ===')
print(' ', hit('NSE-NIFTY-28Apr26-24200-CE',
              '2026-04-23 09:15:00', '2026-04-23 15:30:00'))
time.sleep(0.5)
print(' ', hit('NSE-NIFTY-28Apr26-24200-CE',
              '2026-04-23 09:15:00', '2026-04-23 15:30:00', interval='1minute'))
time.sleep(0.5)

# Test 2: CURRENT/LIVE contract — confirms format works for in-flight options
print('\n=== LIVE: NSE-NIFTY-09Jun26-23500-CE on 2026-06-05 ===')
print(' ', hit('NSE-NIFTY-09Jun26-23500-CE',
              '2026-06-05 09:15:00', '2026-06-05 15:30:00'))
time.sleep(0.5)

# Test 3: Try several other expired contracts to map the recovery surface
print('\n=== Other expired NIFTY weeklies ===')
expired = [
    ('NSE-NIFTY-28Apr26-24200-PE', '2026-04-23 09:15:00', '2026-04-23 15:30:00'),
    ('NSE-NIFTY-30May26-23500-CE', '2026-05-29 09:15:00', '2026-05-29 15:30:00'),  # 2026-05-30 = Sat, but Thu expiry was 28
    ('NSE-NIFTY-28May26-23500-CE', '2026-05-27 09:15:00', '2026-05-27 15:30:00'),
    ('NSE-NIFTY-25Jan24-21500-CE', '2024-01-25 09:15:00', '2024-01-25 15:30:00'),  # active ATM 2 years ago
    ('NSE-NIFTY-25Jan2024-21500-CE', '2024-01-25 09:15:00', '2024-01-25 15:30:00'),  # 4-digit year variant
    ('NSE-NIFTY-25-Jan-24-21500-CE', '2024-01-25 09:15:00', '2024-01-25 15:30:00'),  # extra dashes
]
for sym, s, e in expired:
    print(f'  {sym:38s}  {hit(sym, s, e)}')
    time.sleep(0.5)
