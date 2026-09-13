/* ============================================================
   Arabic Sentiment Analyzer - Frontend Logic
   ============================================================ */

   const chatInner = document.getElementById('chatInner');
   const input = document.getElementById('input');
   const sendBtn = document.getElementById('sendBtn');
   const chatEl = document.getElementById('chat');
   let isBusy = false;
   let currentChat = null;
   
   /* ==================== Theme ==================== */
   function applyTheme(theme) {
       document.documentElement.setAttribute('data-theme', theme);
       const icon = theme === 'light' ? '☀️' : '🌙';
       const label = theme === 'light' ? 'الوضع النهاري' : 'الوضع الليلي';
       document.getElementById('themeIcon').textContent = icon;
       document.getElementById('themeIconTop').textContent = icon;
       document.getElementById('themeLabel').textContent = label;
       localStorage.setItem('theme', theme);
   }
   
   function toggleTheme() {
       const cur = document.documentElement.getAttribute('data-theme') || 'dark';
       applyTheme(cur === 'dark' ? 'light' : 'dark');
   }
   
   /* ==================== Sidebar ==================== */
   function toggleSidebar() {
       document.getElementById('sidebar').classList.toggle('open');
   }
   
   /* ==================== History ==================== */
   function loadHistory() {
       const h = JSON.parse(localStorage.getItem('chats') || '[]');
       const box = document.getElementById('history');
       box.innerHTML = '';
       h.slice().reverse().forEach((c, idx) => {
           const realIdx = h.length - 1 - idx;
           const el = document.createElement('div');
           el.className = 'history-item' + (realIdx === currentChat ? ' active' : '');
           el.textContent = '💬 ' + c.title;
           el.onclick = () => openChat(realIdx);
           box.appendChild(el);
       });
   }
   
   function saveCurrentChat() {
       if (currentChat === null) return;
       const chats = JSON.parse(localStorage.getItem('chats') || '[]');
       if (!chats[currentChat]) return;
       chats[currentChat].html = chatInner.innerHTML;
       localStorage.setItem('chats', JSON.stringify(chats));
       loadHistory();
   }
   
   function newChat() {
       saveCurrentChat();
       currentChat = null;
       chatInner.innerHTML = '';
       chatInner.appendChild(createWelcome());
       document.getElementById('sidebar').classList.remove('open');
       localStorage.removeItem('activeChat');
   }
   
   function openChat(idx) {
       saveCurrentChat();
       const chats = JSON.parse(localStorage.getItem('chats') || '[]');
       if (!chats[idx]) return;
       currentChat = idx;
       chatInner.innerHTML = chats[idx].html;
       document.getElementById('sidebar').classList.remove('open');
       localStorage.setItem('activeChat', idx);
       loadHistory();
       scrollDown();
   }
   
   function createWelcome() {
       const div = document.createElement('div');
       div.className = 'welcome';
       div.id = 'welcome';
       div.innerHTML = `
           <div class="welcome-logo">ع</div>
           <h1>كيف يمكنني مساعدتك؟</h1>
           <p>اكتب أي نص عربي وسأحلل مشاعره فورًا</p>
           <div class="chips">
               <button class="chip" onclick="quickSend('هذا المنتج ممتاز جداً وأنا سعيد بشرائه')">
                   <span class="chip-title">😊 نص إيجابي</span>
                   <span class="chip-sub">تجربة تحليل مشاعر إيجابية</span>
               </button>
               <button class="chip" onclick="quickSend('الخدمة سيئة جداً ولا أنصح بها أبداً')">
                   <span class="chip-title">😞 نص سلبي</span>
                   <span class="chip-sub">تجربة تحليل مشاعر سلبية</span>
               </button>
               <button class="chip" onclick="quickSend('المنتج عادي، لا شيء مميز فيه')">
                   <span class="chip-title">😐 نص محايد</span>
                   <span class="chip-sub">تجربة تحليل مشاعر محايدة</span>
               </button>
               <button class="chip" onclick="quickSend('أحببت هذا التطبيق كثيراً، سريع وسهل الاستخدام!')">
                   <span class="chip-title">⭐ تقييم تطبيق</span>
                   <span class="chip-sub">مثال على مراجعة إيجابية</span>
               </button>
           </div>
       `;
       return div;
   }
   
   /* ==================== Textarea ==================== */
   input.addEventListener('input', () => {
       input.style.height = 'auto';
       input.style.height = Math.min(input.scrollHeight, 200) + 'px';
       sendBtn.disabled = input.value.trim() === '' || isBusy;
   });
   
   input.addEventListener('keydown', (e) => {
       if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
           e.preventDefault();
           send();
       }
   });
   
   /* ==================== Scroll ==================== */
   function scrollDown() {
       setTimeout(() => { chatEl.scrollTop = chatEl.scrollHeight; }, 50);
   }
   
   /* ==================== Messages ==================== */
   function removeWelcome() {
       const w = document.getElementById('welcome');
       if (w) w.remove();
   }
   
   function addUserMessage(text) {
       removeWelcome();
       const el = document.createElement('div');
       el.className = 'msg user';
       el.innerHTML = `
           <div class="avatar">أنا</div>
           <div class="bubble">${escapeHtml(text)}</div>
       `;
       chatInner.appendChild(el);
       scrollDown();
   }
   
   function addBotTyping() {
       const el = document.createElement('div');
       el.className = 'msg bot';
       el.id = 'typingMsg';
       el.innerHTML = `
           <div class="avatar">ع</div>
           <div class="bubble">
               <div class="typing"><span></span><span></span><span></span></div>
           </div>
       `;
       chatInner.appendChild(el);
       scrollDown();
       return el;
   }
   
   function replaceTypingWithResult(data) {
       const typing = document.getElementById('typingMsg');
       if (typing) typing.remove();
   
       const el = document.createElement('div');
       el.className = 'msg bot';
   
       const labelAr = data.label_ar || 'غير معروف';
       let cls = 'neu', emoji = '😐';
       if (labelAr.includes('إيجابي') || labelAr.includes('ايجابي')) {
           cls = 'pos'; emoji = '😊';
       } else if (labelAr.includes('سلبي')) {
           cls = 'neg'; emoji = '😞';
       }
   
       const conf = (data.confidence || 0) * 100;
   
       let probsHtml = '';
       if (data.probabilities && data.probabilities.length === 3) {
           const classes = [
               { name: 'سلبي', color: 'var(--neg)' },
               { name: 'محايد', color: 'var(--neu)' },
               { name: 'إيجابي', color: 'var(--pos)' }
           ];
           data.probabilities.forEach((p, i) => {
               probsHtml += `
                   <div class="prob-row">
                       <span class="prob-name">${classes[i].name}</span>
                       <div class="prob-track">
                           <div class="prob-fill" style="width:${p*100}%; background:${classes[i].color};"></div>
                       </div>
                       <span class="prob-val" style="color:${classes[i].color};">${(p*100).toFixed(1)}%</span>
                   </div>
               `;
           });
       }
   
       const cleanedHtml = data.cleaned
           ? `<div class="cleaned-box"><b>النص بعد المعالجة:</b> ${escapeHtml(data.cleaned)}</div>`
           : '';
   
       el.innerHTML = `
           <div class="avatar">ع</div>
           <div class="bubble">
               <div class="sentiment-card">
                   <div class="sent-head">
                       <div class="sent-emoji">${emoji}</div>
                       <div>
                           <div class="sent-label ${cls}">${labelAr}</div>
                           <div class="sent-conf">نسبة الثقة: ${conf.toFixed(1)}%</div>
                       </div>
                   </div>
                   <div class="probs">${probsHtml}</div>
                   ${cleanedHtml}
               </div>
           </div>
       `;
       chatInner.appendChild(el);
       scrollDown();
   }
   
   function addBotError(message) {
       const typing = document.getElementById('typingMsg');
       if (typing) typing.remove();
       const el = document.createElement('div');
       el.className = 'msg bot';
       el.innerHTML = `
           <div class="avatar">ع</div>
           <div class="bubble" style="border-color: var(--neg);">
               <span style="color: var(--neg);">⚠️ خطأ: ${escapeHtml(message)}</span>
           </div>
       `;
       chatInner.appendChild(el);
       scrollDown();
   }
   
   /* ==================== Send ==================== */
   async function send() {
       const text = input.value.trim();
       if (!text || isBusy) return;
   
       if (currentChat === null) {
           const chats = JSON.parse(localStorage.getItem('chats') || '[]');
           chats.push({ title: text.slice(0, 30), html: '' });
           currentChat = chats.length - 1;
           localStorage.setItem('chats', JSON.stringify(chats));
       }
   
       isBusy = true;
       sendBtn.disabled = true;
       addUserMessage(text);
       input.value = '';
       input.style.height = 'auto';
       addBotTyping();
   
       try {
           const res = await fetch('/predict', {
               method: 'POST',
               headers: { 'Content-Type': 'application/json' },
               body: JSON.stringify({ text })
           });
           const data = await res.json();
           if (!res.ok) throw new Error(data.error || 'فشل التحليل');
           await new Promise(r => setTimeout(r, 250));
           replaceTypingWithResult(data);
       } catch (e) {
           addBotError(e.message);
       } finally {
           isBusy = false;
           sendBtn.disabled = input.value.trim() === '';
           saveCurrentChat();
           input.focus();
       }
   }
   
   function quickSend(text) {
       input.value = text;
       send();
   }
   
   function clearChat() {
       if (!confirm('هل تريد مسح هذه المحادثة؟')) return;
       chatInner.innerHTML = '';
       chatInner.appendChild(createWelcome());
       saveCurrentChat();
   }
   
   function escapeHtml(s) {
       return String(s)
           .replace(/&/g, '&amp;')
           .replace(/</g, '&lt;')
           .replace(/>/g, '&gt;')
           .replace(/"/g, '&quot;')
           .replace(/'/g, '&#39;');
   }
   
   /* ==================== Init ==================== */
   applyTheme(localStorage.getItem('theme') || 'dark');
   
   fetch('/health')
       .then(r => r.json())
       .then(() => {
           document.getElementById('statusText').textContent = 'النموذج جاهز';
           document.getElementById('statusDot').style.background = 'var(--pos)';
       })
       .catch(() => {
           document.getElementById('statusText').textContent = 'الخادم غير متصل';
           document.getElementById('statusDot').style.background = 'var(--neg)';
       });
   
   loadHistory();
   input.focus();