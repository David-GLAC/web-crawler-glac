import streamlit as st
import asyncio
import aiohttp
import json
import re
import io
from datetime import datetime
import time
from urllib.parse import urlparse, urljoin, urlunparse
from collections import defaultdict
from typing import List, Set, Dict, Any
from supabase import create_client, Client
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai import CrawlerRunConfig
from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
from dotenv import load_dotenv

# Streamlit page configuration
st.set_page_config(page_title="Web Crawler App", page_icon="image/web-crawler.png", layout="wide")

# Load custom CSS
def load_css(file_name: str) -> None:
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file {file_name} not found")

load_css("style.css")

# Load environment variables
load_dotenv()

# Initialize Supabase
@st.cache_resource
def init_supabase() -> Client | None:
    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")
    if not url or not key:
        st.error("Missing Supabase credentials")
        return None
    return create_client(url, key)

supabase = init_supabase()

# Authentication functions
def sign_up(email: str, password: str) -> Dict[str, Any]:
    try:
        return supabase.auth.sign_up({"email": email, "password": password})
    except Exception as e:
        return {"error": str(e)}

def sign_in(email: str, password: str) -> Dict[str, Any]:
    try:
        return supabase.auth.sign_in_with_password({"email": email, "password": password})
    except Exception as e:
        return {"error": str(e)}

def sign_out() -> bool:
    try:
        supabase.auth.sign_out()
        return True
    except Exception as e:
        st.error(f"Sign out failed: {str(e)}")
        return False

# Content cleaning and extraction
def clean_content(html: str, url: str) -> str:
    if not html:
        return ""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'noscript', 'svg', 'form', 'button', 'input', 'select']):
            element.decompose()
        for element in soup.find_all(class_=re.compile(r'nav|menu|footer|sidebar|ads|banner', re.I)):
            element.decompose()
        for element in soup.find_all(id=re.compile(r'nav|menu|footer|sidebar|ads|banner', re.I)):
            element.decompose()
        for element in soup.find_all():
            if not element.get_text(strip=True):
                element.decompose()
        text = re.sub(r'\s+', ' ', soup.get_text()).strip()
        return text
    except Exception as e:
        st.warning(f"Content cleaning error for {url}: {str(e)}")
        return html

def extract_main_content(html: str, url: str, css_selector: str = None) -> str:
    """Extract content with proper table formatting when table selector is used"""
    if not html:
        return ""

    try:
        soup = BeautifulSoup(html, 'html.parser')

        # Special handling for table selector
        if css_selector and css_selector.lower().strip() in ["table", "table tr", "table td"]:
            return extract_tables_formatted(soup)

        # Normal CSS selector handling
        if css_selector:
            selected_elements = soup.select(css_selector)
            if selected_elements:
                return "\n".join([
                    element.get_text(" ", strip=True) 
                    for element in selected_elements
                ])

        # Default content extraction
        for element in soup(['script', 'style', 'nav', 'footer']):
            element.decompose()
        return soup.get_text(" ", strip=True)

    except Exception as e:
        print(f"Error extracting content: {str(e)}")
        return ""

def extract_main_content(html: str, url: str, css_selector: str = None) -> str:
    """Extract and clean content with better whitespace handling"""
    if not html:
        return ""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        # 1. CSS Selector mode
        if css_selector:
            selected_elements = soup.select(css_selector)
            if selected_elements:
                content = []
                for idx, element in enumerate(selected_elements):
                    # Check if the element itself is a table
                    if element.name == 'table':
                        # Process the table element directly
                        all_rows = []
                        for row in element.find_all('tr'):
                            row_data = []
                            for cell in row.find_all(['th', 'td']):
                                cell_text = cell.get_text(" ", strip=True)
                                cell_text = clean_whitespace(cell_text)
                                row_data.append(cell_text if cell_text else "-")
                            all_rows.append(row_data)
                        
                        if all_rows:
                            # Calculate max width for each column
                            max_cols = max(len(row) for row in all_rows) if all_rows else 0
                            col_widths = []
                            for col_idx in range(max_cols):
                                max_width = 0
                                for row in all_rows:
                                    if col_idx < len(row):
                                        max_width = max(max_width, len(row[col_idx]))
                                col_widths.append(min(max_width, 30))  # Limit to 30 chars
                            
                            # Format table with proper alignment
                            formatted_rows = []
                            for row_idx, row in enumerate(all_rows):
                                formatted_cells = []
                                for col_idx, cell in enumerate(row):
                                    if col_idx < len(col_widths):
                                        formatted_cells.append(cell.ljust(col_widths[col_idx]))
                                    else:
                                        formatted_cells.append(cell)
                                formatted_rows.append(" | ".join(formatted_cells))
                                
                                # Add separator after header row
                                if row_idx == 0 and len(all_rows) > 1:
                                    separator = " | ".join("-" * width for width in col_widths[:len(row)])
                                    formatted_rows.append(separator)
                            
                            content.append(f"Table {idx+1}:\n" + "\n".join(formatted_rows))
                    else:
                        # Handle tables within selected elements
                        element_tables = []
                        for i, table in enumerate(element.find_all('table')):
                            # Calculate column widths for better formatting
                            all_rows = []
                            for row in table.find_all('tr'):
                                row_data = []
                                for cell in row.find_all(['th', 'td']):
                                    cell_text = cell.get_text(" ", strip=True)
                                    cell_text = clean_whitespace(cell_text)
                                    row_data.append(cell_text if cell_text else "-")
                                all_rows.append(row_data)
                            
                            if all_rows:
                                # Calculate max width for each column
                                max_cols = max(len(row) for row in all_rows) if all_rows else 0
                                col_widths = []
                                for col_idx in range(max_cols):
                                    max_width = 0
                                    for row in all_rows:
                                        if col_idx < len(row):
                                            max_width = max(max_width, len(row[col_idx]))
                                    col_widths.append(min(max_width, 30))  # Limit to 30 chars
                                
                                # Format table with proper alignment
                                formatted_rows = []
                                for row_idx, row in enumerate(all_rows):
                                    formatted_cells = []
                                    for col_idx, cell in enumerate(row):
                                        if col_idx < len(col_widths):
                                            formatted_cells.append(cell.ljust(col_widths[col_idx]))
                                        else:
                                            formatted_cells.append(cell)
                                    formatted_rows.append(" | ".join(formatted_cells))
                                    
                                    # Add separator after header row
                                    if row_idx == 0 and len(all_rows) > 1:
                                        separator = " | ".join("-" * width for width in col_widths[:len(row)])
                                        formatted_rows.append(separator)
                                
                                element_tables.append(f"Table {i+1}:\n" + "\n".join(formatted_rows))
                            
                            # Remove table from element to avoid duplication
                            table.decompose()
                        
                        # Get remaining text content
                        text = element.get_text(" ", strip=True)
                        text = clean_whitespace(text)
                        
                        # Combine text and tables for this element
                        if element_tables:
                            text += "\n\n" + "\n\n".join(element_tables)
                        content.append(text)
                return "\n\n".join(content)
        
        # 2. Full content mode with table preservation
        # Remove unwanted elements but keep tables
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'noscript']):
            element.decompose()
        
        # Special handling for tables with improved formatting
        tables = []
        for i, table in enumerate(soup.find_all('table')):
            # Collect all rows first to calculate column widths
            all_rows = []
            for row in table.find_all('tr'):
                row_data = []
                for cell in row.find_all(['th', 'td']):
                    cell_text = cell.get_text(" ", strip=True)
                    cell_text = clean_whitespace(cell_text)
                    row_data.append(cell_text if cell_text else "-")
                all_rows.append(row_data)
            
            if all_rows:
                # Calculate max width for each column
                max_cols = max(len(row) for row in all_rows) if all_rows else 0
                col_widths = []
                for col_idx in range(max_cols):
                    max_width = 0
                    for row in all_rows:
                        if col_idx < len(row):
                            max_width = max(max_width, len(row[col_idx]))
                    col_widths.append(min(max_width, 30))  # Limit to 30 chars
                
                # Format table with proper alignment
                formatted_rows = []
                for row_idx, row in enumerate(all_rows):
                    formatted_cells = []
                    for col_idx, cell in enumerate(row):
                        if col_idx < len(col_widths):
                            formatted_cells.append(cell.ljust(col_widths[col_idx]))
                        else:
                            formatted_cells.append(cell)
                    formatted_rows.append(" | ".join(formatted_cells))
                    
                    # Add separator after header row
                    if row_idx == 0 and len(all_rows) > 1:
                        separator = " | ".join("-" * width for width in col_widths[:len(row)])
                        formatted_rows.append(separator)
                
                tables.append(f"Table {i+1}:\n" + "\n".join(formatted_rows))
            
            # Remove table from soup to avoid duplication in text content
            table.decompose()
        
        # Get regular text content
        text = soup.get_text(" ", strip=True)
        text = clean_whitespace(text)
        
        # Combine tables and text
        if tables:
            text += "\n\n" + "\n\n".join(tables)
        return text
    except Exception as e:
        print(f"Content extraction error: {str(e)}")
        return clean_whitespace(BeautifulSoup(html, 'html.parser').get_text(" ", strip=True))

def clean_whitespace(text: str) -> str:
    """Normalize whitespace and line breaks"""
    if not text:
        return ""
    
    # Replace any whitespace sequence with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Smart paragraph breaks (after periods followed by capital letters)
    text = re.sub(r'([:!?])\s+([A-Z])', r'\1\n\2', text)
    
    # Remove space before punctuation
    text = re.sub(r'\s+([,:;])', r'\1', text)
    
    # Clean up multiple newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()

# URL utilities
def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc.lower(), parsed.path.rstrip('/'), '', '', ''))

def is_same_domain(url1: str, url2: str) -> bool:
    return urlparse(url1).netloc.lower() == urlparse(url2).netloc.lower()

def is_valid_content_url(url: str) -> bool:
    url_lower = url.lower()
    skip_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp', '.ico',
                       '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.zip', '.rar', '.css',
                       '.mp4', '.mp3', '.avi', '.mov', '.wav', '.flv']
    skip_paths = ['/feed', '/api/', '/admin/', '/wp-admin/', '/assets/', '/static/', '/images/', '/img/', '/css/', '/js/', '/wp-content/', '/administrator/', 'wp-json']
    if any(url_lower.endswith(ext) for ext in skip_extensions) or any(path in url_lower for path in skip_paths):
        return False
    pagination_indicators = ['/page/', '?page=', '&page=', '/p/', '?p=', '&p=']
    return any(indicator in url_lower for indicator in pagination_indicators) or True

def extract_links_from_html(html: str, base_url: str) -> Set[str]:
    links = set()
    href_pattern = r'href=["\']([^"\']+)["\']'
    for match in re.findall(href_pattern, html, re.IGNORECASE):
        absolute_url = urljoin(base_url, match)
        normalized_url = normalize_url(absolute_url)
        if is_same_domain(normalized_url, base_url) and is_valid_content_url(normalized_url):
            links.add(normalized_url)
    return links

# Crawling functions
async def crawl_single_page(session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore, css_selector: str = None) -> Dict[str, Any]:
    async with semaphore:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30), headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }) as response:
                if response.status != 200:
                    return {"url": url, "success": False, "error": f"HTTP {response.status}", "timestamp": datetime.now().isoformat(), "status_code": response.status}
                html = await response.text()
                title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
                title = title_match.group(1).strip() if title_match else "No title"
                meta_desc_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']', html, re.IGNORECASE)
                meta_description = meta_desc_match.group(1).strip() if meta_desc_match else ""
                content = extract_main_content(html, url, css_selector)
                return {
                    "url": url, "title": title, "meta_description": meta_description, "content": content,
                    "html": html, "success": True, "timestamp": datetime.now().isoformat(), "status_code": response.status,
                    "css_selector": css_selector or "None"
                }
        except asyncio.TimeoutError:
            return {"url": url, "success": False, "error": "Timeout after 30 seconds", "timestamp": datetime.now().isoformat()}
        except Exception as e:
            return {"url": url, "success": False, "error": str(e), "timestamp": datetime.now().isoformat()}

async def crawl_links_and_content(start_url: str, max_depth: int = 1, max_pages: int = 50, css_selector: str = None) -> List[Dict[str, Any]]:
    if not start_url.startswith(('http://', 'https://')):
        return [{"success": False, "error": "Invalid URL format", "url": start_url, "timestamp": datetime.now().isoformat()}]
    
    start_time = time.time()    
    crawled_urls, to_crawl, all_results = set(), [normalize_url(start_url)], []
    semaphore = asyncio.Semaphore(min(10, max_pages))
    connector = aiohttp.TCPConnector(limit=min(10, max_pages), limit_per_host=5, keepalive_timeout=40, enable_cleanup_closed=True)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        for depth in range(max_depth):
            if not to_crawl or len(all_results) >= max_pages:
                break
            st.info(f"🕷️ Crawling depth {depth + 1}/{max_depth} - {len(to_crawl)} URLs to process")
            batch_size = min(100, max_pages - len(all_results), len(to_crawl))
            current_batch = [url for url in to_crawl[:batch_size] if url not in crawled_urls]
            to_crawl = to_crawl[batch_size:]
            if not current_batch:
                continue
            
            progress_bar, status_text = st.progress(0), st.empty()
            tasks = [crawl_single_page(session, url, semaphore, css_selector) for url in current_batch]
            completed_results = []
            for i, task in enumerate(asyncio.as_completed(tasks)):
                result = await task
                completed_results.append(result)
                progress_bar.progress((i + 1) / len(tasks))
                status_text.text(f"Processed {i + 1}/{len(tasks)} pages in batch")
                await asyncio.sleep(0.1)
            
            next_urls = []
            for result in completed_results:
                crawled_urls.add(result['url'])
                all_results.append(result)
                if result['success'] and depth < max_depth - 1 and 'html' in result:
                    try:
                        page_links = extract_links_from_html(result['html'], result['url'])
                        new_links = [link for link in page_links if link not in crawled_urls and link not in to_crawl + next_urls]
                        next_urls.extend(new_links)
                        if new_links:
                            st.success(f"📎 Found {len(new_links)} new links from {result['url']}")
                    except Exception as e:
                        st.warning(f"Error extracting links from {result['url']}: {str(e)}")
            
            if next_urls:
                to_crawl.extend(next_urls)
                st.info(f"✅ Added {len(next_urls)} new URLs for next depth")
            else:
                st.warning("No new URLs found for next depth")
            
            progress_bar.empty()
            status_text.empty()
            for result in completed_results:
                result.pop('html', None)
    
    elapsed_time = time.time() - start_time
    successful = sum(1 for r in all_results if r.get('success', False))
    st.success(f"✅ Crawling completed in {elapsed_time:.2f} seconds 🍾")
    st.info(f"📊 Results: {successful} successful + {len(all_results) - successful} failed = {len(all_results)} total pages")
    st.info(f"🔍 Unique URLs discovered: {len(crawled_urls)}")
    return all_results

async def crawl_single_url(url: str, css_selector: str = None) -> Dict[str, Any]:
    """Crawl a single URL using aiohttp for consistency with tab1"""
    semaphore = asyncio.Semaphore(1)  # Single request for simplicity
    connector = aiohttp.TCPConnector(limit=1, keepalive_timeout=40)
    async with aiohttp.ClientSession(connector=connector) as session:
        result = await crawl_single_page(session, url, semaphore, css_selector)
        return result

async def crawl_urls_with_selector(urls: List[str], css_selector: str = None) -> List[Dict[str, Any]]:
    """Crawl multiple URLs with enhanced error handling"""
    if not urls:
        return [{"success": False, "url": "", "error": "No URLs provided", "timestamp": datetime.now().isoformat()}]
    
    results = []
    max_concurrent = min(10, len(urls))
    semaphore = asyncio.Semaphore(max_concurrent)
    connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=5, keepalive_timeout=40)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        progress_bar, status_text = st.progress(0), st.empty()
        tasks = [crawl_single_page(session, url, semaphore, css_selector) for url in urls]
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            results.append(result)
            progress_bar.progress((i + 1) / len(tasks))
            status_text.text(f"Processed {i + 1}/{len(tasks)} URLs")
            await asyncio.sleep(0.1)
        progress_bar.empty()
        status_text.empty()
    
    return results

# Database operations
def save_crawl_result(user_id: str, crawl_data: Dict[str, Any]) -> None:
    try:
        existing = supabase.table('crawl_results').select('id').eq('user_id', user_id).eq('url', crawl_data["url"]).execute()
        data = {
            "title": crawl_data.get("title", ""), "content": crawl_data.get("content", ""),
            "meta_description": crawl_data.get("meta_description", ""), "links": json.dumps(crawl_data.get("links", [])),
            "success": crawl_data["success"], "error_message": crawl_data.get("error", ""), "created_at": crawl_data["timestamp"]
        }
        if existing.data:
            supabase.table('crawl_results').update(data).eq('id', existing.data[0]['id']).execute()
        else:
            data.update({"user_id": user_id, "url": crawl_data["url"]})
            supabase.table('crawl_results').insert(data).execute()
    except Exception as e:
        st.error(f"Error saving crawl result: {str(e)}")

def get_user_crawl_history(user_id: str) -> List[Dict[str, Any]]:
    try:
        return supabase.table('crawl_results').select("*").eq('user_id', user_id).order('created_at', desc=True).execute().data
    except Exception as e:
        st.error(f"Error fetching crawl history: {str(e)}")
        return []

def delete_crawl_result(result_id: int) -> bool:
    try:
        supabase.table('crawl_results').delete().eq('id', result_id).execute()
        return True
    except Exception as e:
        st.error(f"Error deleting crawl result: {str(e)}")
        return False

def clear_all_history(user_id: str) -> bool:
    try:
        supabase.table('crawl_results').delete().eq('user_id', user_id).execute()
        return True
    except Exception as e:
        st.error(f"Error clearing history: {str(e)}")
        return False

# Streamlit UI
def init_session_state() -> None:
    defaults = {
        'authenticated': False, 'user': None, 'url_crawl_results': [], 'is_crawling': False,
        'crawl_results': None, 'last_crawled_url': ""
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def show_auth_page() -> None:
    st.image("image/web-crawler.png")
    st.title("Web Crawler Authentication")
    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])
    
    with tab1:
        st.subheader("Sign In")
        with st.form("signin_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Sign In"):
                if email and password:
                    with st.spinner("Signing in..."):
                        result = sign_in(email, password)
                        if "error" not in result:
                            st.session_state.authenticated = True
                            st.session_state.user = result.user
                            st.success("🍾 Signed in successfully!")
                            st.rerun()
                        else:
                            st.error(f"Sign in failed: {result['error']}")
                else:
                    st.error("Please fill in all fields")
    
    with tab2:
        st.subheader("Sign Up")
        with st.form("signup_form"):
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password")
            if st.form_submit_button("Sign Up"):
                if email and password and confirm_password:
                    if password == confirm_password:
                        with st.spinner("Creating account..."):
                            result = sign_up(email, password)
                            if "error" not in result:
                                st.success("🍾 Account created successfully! Please check your email for verification.")
                            else:
                                st.error(f"Sign up failed: {result['error']}")
                    else:
                        st.error("Passwords do not match")
                else:
                    st.error("Please fill in all fields")

def display_results(results: List[Dict[str, Any]], source: str = "crawl") -> None:
    successful = [r for r in results if r.get('success', False)]
    if not successful:
        st.error("No URLs crawled successfully")
        return
    
    combined_content = "\n\n".join(
        f"=== URL: {r['url']} ===\nTitle: {r.get('title', 'No title')}\nMeta Description: {r.get('meta_description', 'No description')}\nContent:\n{r.get('content', 'No content')}\n"
        for r in successful
    )
    
    st.success(f"🍾 Successfully crawled {len(successful)}/{len(results)} URLs")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Pages", len(results))
    col2.metric("✅ Successful", len(successful))
    col3.metric("❌ Failed", len(results) - len(successful))
    col4.metric("📄 Content Size", f"{len(combined_content):,} chars")
    
    with st.expander("👀 Content Preview"):
        preview_length = 3000
        preview_text = combined_content[:preview_length] + (
            f"\n\n... [Showing first {preview_length} characters of {len(combined_content)} total]"
            if len(combined_content) > preview_length else ""
        )
        st.text_area("Content Preview", preview_text, height=300)
    
    with st.expander("🔗 Detailed URL Status"):
        for i, result in enumerate(results, 1):
            if result.get('success', False):
                st.success(f"{i}. ✅ {result['url']}")
                if result.get('title'):
                    st.write(f"   📝 {result['title']}")
            else:
                st.error(f"{i}. ❌ {result['url']}")
                if result.get('error'):
                    st.write(f"   🚫 Error: {result['error']}")
    
    st.subheader("⬇️ Download Options")
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    
    with col_dl1:
        st.download_button(
            label="📄 Download All (TXT)",
            data=combined_content,
            file_name=f"{source}_content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    with col_dl2:
        json_data = json.dumps([{
            "url": r['url'], "title": r.get('title'), "content": r.get('content'),
            "meta_description": r.get('meta_description'), "css_selector": r.get('css_selector')
        } for r in successful], indent=2, ensure_ascii=False)
        st.download_button(
            label="📊 Download as JSON",
            data=json_data,
            file_name=f"{source}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col_dl3:
        try:
            pdfmetrics.registerFont(TTFont('DejaVuSans', 'fonts/DejaVuSans.ttf'))
        except:
            st.warning("Font file not found, using default font")
        
        def generate_pdf():
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            vn_style = ParagraphStyle('Vietnamese', parent=styles['Normal'], fontName='DejaVuSans', fontSize=10, leading=12, alignment=TA_LEFT)
            story = []
            for item in successful:
                story.extend([
                    Paragraph(f"<b>URL:</b> {item['url']}", vn_style),
                    Paragraph(f"<b>Title:</b> {item.get('title', 'No title')}", vn_style),
                    Paragraph(f"<b>Meta Description:</b> {item.get('meta_description', 'No description')}", vn_style),
                    Paragraph("<b>Content:</b>", vn_style),
                    Paragraph(item.get('content', 'No content')[:20000] + ("... [Content truncated]" if len(item.get('content', '')) > 20000 else ""), vn_style),
                    Spacer(1, 12)
                ])
            doc.build(story)
            buffer.seek(0)
            return buffer
        
        pdf_buffer = generate_pdf()
        st.download_button(
            label="📑 Download as PDF",
            data=pdf_buffer,
            file_name=f"{source}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )

def show_crawl_interface() -> None:
    st.info(
        "📌 With this feature, input a domain (e.g., `https://example.com`) to:\n"
        "- Automatically scan and collect internal links.\n"
        "- Extract content, titles, and metadata from each page.\n"
        "- Filter by URL patterns or keywords, and set page limits.\n"
        "- Ideal for collecting content from an entire website."
    )
    
    url = st.text_input("Website URL to crawl:", placeholder="🔗 https://example.com")
    with st.expander("🚀 Advanced Options"):
        col1, col2 = st.columns([8,4])
        with col1:
            max_depth = st.number_input("Max crawl depth", min_value=1, max_value=6, value=1, step=1)
            max_pages = st.number_input("Max pages to crawl", min_value=1, max_value=3000, value=50, step=50)
            css_selector = st.text_input("✏️ Attribute Selector (optional):", placeholder=".article-content, #main, table, etc.")
        with col2:
            st.info(
                "⚡️ Features:\n- Same domain only\n- Excludes images & documents\n- Auto deduplication\n"
                "- Rate limiting\n- Memory optimization\n- Collect optional attributes\n- Normalize to clean text"
            )
    
    if st.button("🚀 Start Enhanced Crawler", type="primary") and url:
        if not url.startswith(('http://', 'https://')):
            st.warning("⚠️ Please enter a valid URL starting with http:// or https://")
            return
        if max_pages > 100:
            st.warning(f"⚠️ You're about to crawl {max_pages} pages. This may take several minutes.")
        with st.spinner("🕷️ Enhanced crawling in progress..."):
            results = asyncio.run(crawl_links_and_content(url, max_depth, max_pages, css_selector))
            st.session_state.crawl_results = results
            st.session_state.last_crawled_url = url
            if results and any(r.get('success', False) for r in results):
                st.success("✔️ Enhanced crawling completed! 🍾")
                st.balloons()
                if st.session_state.user:
                    for result in results:
                        save_crawl_result(st.session_state.user.id, result)
                display_results(results, "crawl")

def show_crawl_url() -> None:
    st.info(
        "📌 This feature lets you:\n"
        "- Paste or upload a list of URLs to extract data from.\n"
        "- Crawl specific pages for body text, titles, and metadata.\n"
        "- Perfect for targeted content collection from known URLs."
    )
    
    with st.form("url_crawl_form"):
       
    
        url = st.text_input("Enter URL:", placeholder="🔗 https://example.com")
        uploaded_file = st.file_uploader("Or upload URL list (TXT)", type=["txt"])
        css_selector = st.text_input("✏️ Attribute Selector (optional):", placeholder=".article-content, #main, table, etc.")
        submit_button = st.form_submit_button("🚀 Start Crawling", type="primary")
    
    if submit_button:
        urls = []
        if uploaded_file:
            urls = [line.strip() for line in uploaded_file.read().decode("utf-8").splitlines() if line.strip()]
        elif url:
            urls = [url.strip()]
        
        if not urls:
            st.error("Please provide at least one valid URL")
            return
        
        # Validate URLs
        valid_urls = []
        for u in urls:
            if not u.startswith(('http://', 'https://')):
                st.warning(f"Invalid URL skipped: {u} (must start with http:// or https://)")
                continue
            valid_urls.append(u)
        
        if not valid_urls:
            st.error("No valid URLs provided")
            return
        
        if uploaded_file:
            st.success(f"📁 Loaded {len(valid_urls)} valid URLs from file")
        
        with st.spinner(f"🕷️ Crawling {len(valid_urls)} URLs..."):
            results = asyncio.run(crawl_urls_with_selector(valid_urls, css_selector))
            st.session_state.url_crawl_results = results
            if st.session_state.user:
                for result in results:
                    save_crawl_result(st.session_state.user.id, result)
            if any(r.get('success', False) for r in results):
                st.success("✔️ Crawling completed!")
                st.balloons()
                display_results(results, "url_crawl")
            else:
                st.error("No URLs crawled successfully. Check detailed errors below.")
                with st.expander("🔍 Detailed Errors"):
                    for i, result in enumerate(results, 1):
                        st.error(f"{i}. ❌ {result['url']}: {result.get('error', 'Unknown error')}")

def show_crawl_history() -> None:
    st.info("View and manage your crawled content history.")
    if not st.session_state.user:
        st.error("Please login to view crawl history")
        return
    
    history = get_user_crawl_history(st.session_state.user.id)
    if not history:
        st.info("No crawl history found. Start crawling some websites!")
        return
    
    if st.button("🗑️ Clear All History", type="secondary"):
        if clear_all_history(st.session_state.user.id):
            st.success("All history cleared!")
            st.rerun()
    
    domains = defaultdict(list)
    for item in history:
        domains[urlparse(item['url']).netloc].append(item)
    
    for domain_idx, (domain, items) in enumerate(domains.items()):
        with st.expander(f"🌐 {domain} - {len(items)} pages"):
            successful = [item for item in items if item['success']]
            combined_content = "\n\n".join(
                f"=== {item['url']} ===\nTitle: {item.get('title', 'No title')}\nDate: {item['created_at'][:19]}\n\n{item.get('content', 'No content')}"
                for item in successful
            )
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            col_stat1.metric("Total Pages", len(items))
            col_stat2.metric("✅ Successful", len(successful))
            col_stat3.metric("❌ Failed", len(items) - len(successful))
            
            st.download_button(
                label="⬇️ Download Content .txt",
                data=combined_content,
                file_name=f"crawl_{domain}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                key=f"download_{domain_idx}"
            )
            
            for item_idx, item in enumerate(items):
                with st.container():
                    col1, col2 = st.columns([9,3])
                    with col1:
                        st.write(f"**URL:** {item['url']}")
                        st.write(f"**Date:** {item['created_at'][:19]}")
                        if not item['success']:
                            st.error(f"Error: {item.get('error_message', 'Unknown error')}")
                
                    with col2:
                        if st.button("❌ Delete", key=f"delete_{domain_idx}_{item_idx}"):
                            if delete_crawl_result(item['id']):
                                st.success("Item deleted!")
                                st.rerun()

def handle_recrawl(url: str) -> None:
    with st.spinner(f"Re-crawling {url}..."):
        result = asyncio.run(crawl_single_url(url))
        if result["success"] and st.session_state.user:
            save_crawl_result(st.session_state.user.id, result)
            st.success("🍾 Re-crawled successfully!")
            st.rerun()

def main() -> None:
    init_session_state()
    if supabase and supabase.auth.get_user():
        st.session_state.authenticated = True
        st.session_state.user = supabase.auth.get_user().user
    
    if not st.session_state.authenticated:
        show_auth_page()
    else:
        if st.button("Sign Out"):
            sign_out()
            st.session_state.authenticated = False
            st.session_state.user = None
            st.rerun()
        if st.session_state.user:
            st.info(f"Welcome, {st.session_state.user.email} 👋")
        st.title("Web Crawler & Scraper 🚀")
        tab1, tab2, tab3 = st.tabs(["🌐 Auto Crawl Website by Domain", "📄 Crawl Specific URLs", "📊 Crawl History"])
        with tab1:
            show_crawl_interface()
        with tab2:
            show_crawl_url()
        with tab3:
            show_crawl_history()

if __name__ == "__main__":
    main()