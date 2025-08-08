const fs = require('fs');
const path = require('path');
const vm = require('node:vm');
const test = require('node:test');
const assert = require('node:assert/strict');
const { JSDOM } = require('jsdom');

const htmlPath = path.join(__dirname, 'admin.html');
const html = fs.readFileSync(htmlPath, 'utf8');
const scriptContent = html.match(/<script>([\s\S]*)<\/script>/)[1];

function setupDom() {
  const dom = new JSDOM(
    `<select id="tenantName"></select>
<div id="newTenantNameGroup" class="hidden"><input id="newTenantName" /></div>
<div id="createTenantModal" class="hidden"></div>`,
    { url: 'http://localhost', runScripts: 'dangerously' }
  );

  // Ignore DOMContentLoaded handlers to prevent unrelated errors
  const originalAddEventListener = dom.window.document.addEventListener.bind(dom.window.document);
  dom.window.document.addEventListener = (type, listener, options) => {
    if (type !== 'DOMContentLoaded') {
      originalAddEventListener(type, listener, options);
    }
  };

  dom.window.fetch = async () => ({
    ok: true,
    json: async () => [{ tenant: 'TenantA' }, { tenant: 'TenantB' }]
  });

  const scriptEl = dom.window.document.createElement('script');
  scriptEl.textContent = scriptContent;
  dom.window.document.body.appendChild(scriptEl);

  return dom;
}

test('limited access user sees only allowed tenants', async () => {
  const dom = setupDom();
  const vmCtx = dom.getInternalVMContext();
  vm.runInContext('currentUser = { tenant: "TenantA", role: "user" };', vmCtx);

  await dom.window.loadTenantsForCreation();

  const select = dom.window.document.getElementById('tenantName');
  const options = Array.from(select.options).map(o => o.value);
  assert.deepStrictEqual(options, ['TenantA', 'TenantB']);

  const group = dom.window.document.getElementById('newTenantNameGroup');
  const input = dom.window.document.getElementById('newTenantName');
  assert(group.classList.contains('hidden'));
  assert.equal(input.required, false);
});

test('all access user can create new tenant and toggles new tenant input', async () => {
  const dom = setupDom();
  const vmCtx = dom.getInternalVMContext();
  vm.runInContext('currentUser = { tenant: "*", role: "system_admin" };', vmCtx);

  await dom.window.loadTenantsForCreation();

  const select = dom.window.document.getElementById('tenantName');
  const options = Array.from(select.options).map(o => o.value);
  assert.deepStrictEqual(options, ['TenantA', 'TenantB', '__new__']);

  const group = dom.window.document.getElementById('newTenantNameGroup');
  const input = dom.window.document.getElementById('newTenantName');
  assert(group.classList.contains('hidden'));
  assert.equal(input.required, false);

  select.value = '__new__';
  select.dispatchEvent(new dom.window.Event('change'));
  assert(!group.classList.contains('hidden'));
  assert.equal(input.required, true);

  select.value = 'TenantA';
  select.dispatchEvent(new dom.window.Event('change'));
  assert(group.classList.contains('hidden'));
  assert.equal(input.required, false);
});
