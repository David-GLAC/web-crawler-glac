# app.py
import streamlit as st
import asyncio
from supabase import create_client, Client
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.async_configs import BrowserConfig
import os
from dotenv import load_dotenv
import json
from datetime import datetime
from bs4 import BeautifulSoup
import re
from collections import defaultdict
from typing import List, Set, Dict, Any
import time
from urllib.parse import urljoin, urlparse, urlunparse
import aiohttp






# MUST BE FIRST: Configure Streamlit page
st.set_page_config(
    page_title="Web Crawler App",
    page_icon="image/web-crawler.png",
    layout="wide"
)
# Load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")
# Load environment variables
load_dotenv()

# Initialize Supabase
@st.cache_resource
def init_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        st.error("Missing Supabase credentials in environment variables")
        return None
    return create_client(url, key)

supabase = init_supabase()

# Authentication functions
def sign_up(email, password):
    try:
        response = supabase.auth.sign_up({
            "email": email,
            "password": password
        })
        return response
    except Exception as e:
        return {"error": str(e)}

def sign_in(email, password):
    try:
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        return response
    except Exception as e:
        return {"error": str(e)}

def sign_out():
    try:
        supabase.auth.sign_out()
        return True
    except Exception as e:
        return False

# Content cleaning functions
def clean_content(content, url):
    """Clean and normalize crawled content using BeautifulSoup"""
    if not content:
        return ""
    
    try:
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 
                           'iframe', 'noscript', 'svg', 
                           'form', 'button', 'input', 'select']):
            element.decompose()
        
        # Remove common classes/ids
        for element in soup.find_all(class_=re.compile(r'nav|menu|footer|sidebar|ads|banner', re.I)):
            element.decompose()
            
        for element in soup.find_all(id=re.compile(r'nav|menu|footer|sidebar|ads|banner', re.I)):
            element.decompose()
        
        # Remove empty elements
        for element in soup.find_all():
            if len(element.get_text(strip=True)) == 0:
                element.decompose()
        
        # Normalize whitespace
        text = soup.get_text()
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    except Exception as e:
        st.warning(f"Content cleaning error for {url}: {str(e)}")
        return content

def extract_main_content(html, url):
    """Extract main content using heuristics"""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Try to find article or main tags first
        article = soup.find('article') or soup.find('main')
        if article:
            return clean_content(str(article), url)
        
        # Fallback to content-heavy sections
        candidates = []
        for elem in soup.find_all(['div', 'section']):
            text = elem.get_text()
            text_length = len(text.strip())
            link_density = len(elem.find_all('a')) / (text_length / 100) if text_length > 0 else 0
            
            # Score based on text length and link density
            if text_length > 200 and link_density < 2:
                candidates.append((elem, text_length))
        
        if candidates:
            # Get the element with most text
            main_elem = max(candidates, key=lambda x: x[1])[0]
            return clean_content(str(main_elem), url)
        
        # Final fallback to cleaned body
        return clean_content(html, url)
        
    except Exception as e:
        st.warning(f"Main content extraction error for {url}: {str(e)}")
        return clean_content(html, url)

def normalize_url(url):
    """Normalize URL for comparison"""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"

# Crawling functions
async def crawl_url(url):
    """Crawl content from a given URL using Crawl4AI with improved content extraction"""
    try:
        # Configure browser for Docker environment
        browser_config = {
            "headless": True,
            "verbose": True,
            "browser_type": "chromium",
            "chrome_channel": "chrome",
            "user_agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        async with AsyncWebCrawler(**browser_config) as crawler:
            result = await crawler.arun(
                url=url,
                word_count_threshold=10,
                extraction_strategy="NoExtractionStrategy",
                chunking_strategy="RegexChunking"
            )
            
            # Extract and clean content
            raw_content = result.html if hasattr(result, 'html') else ""
            cleaned_content = extract_main_content(raw_content, url)
            
            meta_description = ""
            if result.metadata:
                meta_description = result.metadata.get("description", "")
                
            # Get unique links (normalized)
            raw_links = result.links if hasattr(result, 'links') and result.links else []
            unique_links = list({normalize_url(link) for link in raw_links})
            
            return {
                "success": True,
                "url": url,
                "title": result.metadata.get("title", "No title") if result.metadata else "No title",
                "meta_description": meta_description,
                "content": cleaned_content if cleaned_content else "No content extracted",
                "links": unique_links,
                "timestamp": datetime.now().isoformat(),
                "raw_content": raw_content  # Store raw for debugging
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "url": url,
            "timestamp": datetime.now().isoformat()
        }

async def crawl_links_and_content(start_url, max_depth=1, max_pages=5):
    """Crawl both links and their content with improved deduplication"""
    try:
        link_config = CrawlerRunConfig(
            deep_crawl_strategy=BFSDeepCrawlStrategy(
                max_depth=max_depth,
                max_pages=max_pages,
                include_external=False                   
            ),
            verbose=True
        )
        
        async with AsyncWebCrawler() as crawler:
            # Get links with deduplication
            link_results = await crawler.arun(start_url, config=link_config)
            unique_urls = list({normalize_url(result.url) for result in link_results})
            
            # Crawl content for each unique URL
            content_results = []
            content_config = CrawlerRunConfig()            
            
            for url in unique_urls[:max_pages]:  # Respect max_pages limit
                try:
                    result = await crawler.arun(
                        url=url,
                        config=content_config
                    )
                    
                    # Extract and clean content
                    raw_content = result.html if hasattr(result, 'html') else ""
                    cleaned_content = extract_main_content(raw_content, url)
                    
                    meta_description = ""
                    if result.metadata:
                        meta_description = result.metadata.get("description", "")
                        
                    content_results.append({
                        "url": url,
                        "title": result.metadata.get("title", "No title") if result.metadata else "No title",
                        "meta_description": meta_description,
                        "content": cleaned_content,
                        "success": True,
                        "timestamp": datetime.now().isoformat(),
                        "raw_content": raw_content  # Store raw for debugging
                    })
                except Exception as e:
                    content_results.append({
                        "url": url,
                        "success": False,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
                    
            return content_results
    except Exception as e:
        return [{
            "success": False,
            "error": str(e),
            "url": start_url,
            "timestamp": datetime.now().isoformat()
        }]

def save_crawl_result(user_id, crawl_data):
    """Save crawl result to Supabase with update if exists"""
    try:
        # Check if URL already exists
        existing = supabase.table('crawl_results')\
                         .select('id')\
                         .eq('user_id', user_id)\
                         .eq('url', crawl_data["url"])\
                         .execute()
        
        if existing.data:
            # Update existing record
            response = supabase.table('crawl_results').update({
                "title": crawl_data.get("title", ""),
                "content": crawl_data.get("content", ""),
                "meta_description": crawl_data.get("meta_description", ""),
                "links": json.dumps(crawl_data.get("links", [])),
                "success": crawl_data["success"],
                "error_message": crawl_data.get("error", ""),
                "created_at": crawl_data["timestamp"]
            }).eq('id', existing.data[0]['id']).execute()
        else:
            # Insert new record
            response = supabase.table('crawl_results').insert({
                "user_id": user_id,
                "url": crawl_data["url"],
                "title": crawl_data.get("title", ""),
                "meta_description": crawl_data.get("meta_description", ""),
                "content": crawl_data.get("content", ""),
                "links": json.dumps(crawl_data.get("links", [])),
                "success": crawl_data["success"],
                "error_message": crawl_data.get("error", ""),
                "created_at": crawl_data["timestamp"]
            }).execute()
        
        return response
    except Exception as e:
        st.error(f"Error saving crawl result: {str(e)}")
        return None

def get_user_crawl_history(user_id):
    """Get crawl history for a user"""
    try:
        response = supabase.table('crawl_results').select("*").eq('user_id', user_id).order('created_at', desc=True).execute()
        return response.data
    except Exception as e:
        st.error(f"Error fetching crawl history: {str(e)}")
        return []

def delete_crawl_result(result_id):
    """Delete a specific crawl result"""
    try:
        response = supabase.table('crawl_results').delete().eq('id', result_id).execute()
        return True
    except Exception as e:
        st.error(f"Error deleting crawl result: {str(e)}")
        return False

def clear_all_history(user_id):
    """Delete all history for a user"""
    try:
        response = supabase.table('crawl_results').delete().eq('user_id', user_id).execute()
        return True
    except Exception as e:
        st.error(f"Error clearing history: {str(e)}")
        return False

# Streamlit UI
def main():
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'url_crawl_results' not in st.session_state:
        st.session_state.url_crawl_results = []
    if 'is_crawling' not in st.session_state:
        st.session_state.is_crawling = False
    if 'crawl_results' not in st.session_state:
        st.session_state.crawl_results = None
    if 'last_crawled_url' not in st.session_state:
        st.session_state.last_crawled_url = ""
    
    # Check if user is already authenticated
    if supabase and supabase.auth.get_user():
        st.session_state.authenticated = True
        st.session_state.user = supabase.auth.get_user().user
    
    if not st.session_state.authenticated:
        show_auth_page()
    else:
        show_main_app()

def show_auth_page():
    st.image("image/web-crawler.png")
    st.title("Web Crawler Authentication")
    
    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])
    
    with tab1:
        st.subheader("Sign In")
        with st.form("signin_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Sign In")
            
            if submit:
                if email and password:
                    with st.spinner("Signing in..."):
                        result = sign_in(email, password)
                        if "error" not in result:
                            st.session_state.authenticated = True
                            st.session_state.user = result.user
                            st.success("Signed in successfully!")
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
            submit = st.form_submit_button("Sign Up")
            
            if submit:
                if email and password and confirm_password:
                    if password == confirm_password:
                        with st.spinner("Creating account..."):
                            result = sign_up(email, password)
                            if "error" not in result:
                                st.success("Account created successfully! Please check your email for verification.")
                            else:
                                st.error(f"Sign up failed: {result['error']}")
                    else:
                        st.error("Passwords do not match")
                else:
                    st.error("Please fill in all fields")

def show_main_app():
    # Header
    if st.button("Sign Out"):
            sign_out()
            st.session_state.authenticated = False
            st.session_state.user = None
            st.rerun()

        # User info
    if st.session_state.user:
        st.info(f"Welcome, {st.session_state.user.email} 👋")

    st.title("Web Crawler & Scraper 🚀")

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌐 Crawl Website",
        "📄 Crawl Full Content (URLs)",
        "🎯 Targeted Content Crawl (Attribute) ",
        "📊 Crawl History"
    ])

    with tab1:
        show_crawl_interface()
    
    with tab2:
        show_crawl_url()
    with tab3:
        show_crawl_selector()

    with tab4:
        show_crawl_history()     

def show_crawl_selector():
    st.info("Extract specific elements such as `<title>`, meta description, keywords, headings, or custom CSS selectors.")

def show_crawl_url():
    st.info("Enter a single URL or upload a `.txt` file containing multiple URLs to extract the full page content.")

    # Input form
    with st.form("url_crawl_form"):
        url = st.text_input("Enter URL:", placeholder="🔗 https://example.com")
        uploaded_file = st.file_uploader("Or upload a TXT file with URLs", type=["txt"])
        submit_button = st.form_submit_button("Start Crawling", type="primary")

    # Handle form submission
    if submit_button and (url or uploaded_file):
        st.session_state.is_crawling = True
        st.session_state.url_crawl_results = []
        
        urls_to_crawl = []
        
        if uploaded_file is not None:
            file_content = uploaded_file.read().decode("utf-8")
            urls_to_crawl = [line.strip() for line in file_content.splitlines() if line.strip()]
            st.success(f"Loaded {len(urls_to_crawl)} URLs from file")
        elif url:
            urls_to_crawl = [url.strip()]
        
        if urls_to_crawl:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Crawl each URL
            for i, url in enumerate(urls_to_crawl):
                try:
                    status_text.text(f"⚡️ Crawling {i+1}/{len(urls_to_crawl)}: {url}")
                    progress_bar.progress((i+1)/len(urls_to_crawl))
                    
                    # Perform crawling
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(crawl_url(url))
                    loop.close()
                    
                    # Save results
                    st.session_state.url_crawl_results.append(result)
                    
                    # Save to database
                    if st.session_state.user:
                        save_crawl_result(st.session_state.user.id, result)
                        
                except Exception as e:
                    st.session_state.url_crawl_results.append({
                        "success": False,
                        "url": url,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
            
            st.session_state.is_crawling = False
            progress_bar.empty()
            success_count = len([r for r in st.session_state.url_crawl_results if r['success']])
            
            # Final success notification
            if success_count > 0:
                st.balloons()
                st.success(f"🎉 Successfully crawled {success_count}/{len(urls_to_crawl)} URLs!")
            else:
                st.error("⚠️ Crawling completed but no URLs were successful")
            
            st.rerun()

    # Show results and download options
    if st.session_state.url_crawl_results and not st.session_state.is_crawling:
        results = st.session_state.url_crawl_results
        successful = [r for r in results if r['success']]
        
        if successful:
            # Prepare combined data
            combined_txt = "\n\n".join(
                f"=== URL: {r['url']} ===\n"
                f"Title: {r.get('title', 'No title')}\n"
                f"Meta Description: {r.get('meta_description', 'No description')}\n"
                f"Content:\n{r.get('content', 'No content')}"
                for r in successful
            )
            
            combined_json = {
                "metadata": {
                    "total_urls": len(results),
                    "successful": len(successful),
                    "timestamp": datetime.now().isoformat()
                },
                "results": [
                    {
                        "url": r['url'],
                        "title": r.get('title'),
                        "meta_description": r.get('meta_description'),
                        "content": r.get('content'),
                        "timestamp": r['timestamp']
                    } for r in successful
                ]
            }
            
            # Preview combined content
            st.success(f"🎉 Successfully crawled {len(successful)} URLs")
            with st.expander("👀 Preview content...", expanded=False):
                st.write("**Meta Descriptions:**")
                for r in successful:
                    st.write(f"- {r['url']}: {r.get('meta_description', 'No description')}")
                st.text_area("Content .txt", 
                            combined_txt[:5000] + "..." if len(combined_txt) > 5000 else combined_txt,
                            height=400,
                            key="combined_preview")
            
            st.markdown("### Download Options")
            
            # Format selection
            download_format = st.radio(
                "Select format:",
                ["TXT", "JSON", "PDF"],
                horizontal=True,
                label_visibility="collapsed"
            )
            
            # Download buttons
            if download_format == "TXT":
                st.download_button(
                    label="Download",
                    type="primary",
                    data=combined_txt,
                    file_name=f"crawled_content_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    key="dl_txt"
                )
            
            elif download_format == "JSON":
                st.download_button(
                    label="Download",
                    type="primary",
                    data=json.dumps(combined_json, indent=2, ensure_ascii=False),
                    file_name=f"crawled_content_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    key="dl_json"
                )
            
            elif download_format == "PDF":
                from reportlab.lib.pagesizes import letter
                from reportlab.pdfbase import pdfmetrics
                from reportlab.pdfbase.ttfonts import TTFont
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.enums import TA_LEFT
                pdfmetrics.registerFont(TTFont('DejaVuSans', 'fonts/DejaVuSans.ttf'))
                
                import io
                
                def generate_pdf():
                    buffer = io.BytesIO()
                    doc = SimpleDocTemplate(buffer, pagesize=letter)
                    styles = getSampleStyleSheet()
                    vn_style = ParagraphStyle(
                        'Vietnamese',
                        parent=styles['Normal'],
                        fontName='DejaVuSans',
                        fontSize=10,
                        leading=12,
                        alignment=TA_LEFT
                    )
                    story = []
                    
                    for item in successful:
                        story.append(Paragraph(f"<b>URL:</b> {item['url']}", vn_style))
                        story.append(Paragraph(f"<b>Title:</b> {item.get('title', 'No title')}", vn_style))
                        story.append(Paragraph(f"<b>Meta Description:</b> {item.get('meta_description', 'No description')}", vn_style))
                        story.append(Paragraph("<b>Content:</b>", vn_style))
                        story.append(Paragraph(item.get('content', 'No content'), vn_style))
                        story.append(Spacer(1, 12))
                    
                    doc.build(story)
                    buffer.seek(0)
                    return buffer
                
                pdf_buffer = generate_pdf()
                st.download_button(
                    label="Download",
                    type="primary",
                    data=pdf_buffer,
                    file_name=f"crawled_content_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    key="dl_pdf"
                )
def normalize_url(url: str) -> str:
    """Normalize URL by removing fragments and query params"""
    parsed = urlparse(url)
    normalized = urlunparse((
        parsed.scheme,
        parsed.netloc.lower(),
        parsed.path.rstrip('/'),
        '',  # Remove params
        '',  # Remove query
        ''   # Remove fragment
    ))
    return normalized

def is_same_domain(url1: str, url2: str) -> bool:
    """Check if two URLs belong to the same domain"""
    domain1 = urlparse(url1).netloc.lower()
    domain2 = urlparse(url2).netloc.lower()
    return domain1 == domain2

def is_valid_content_url(url: str) -> bool:
    """Filter out image, document, and other non-content URLs"""
    url_lower = url.lower()
    
    # Skip image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp', '.ico']
    if any(url_lower.endswith(ext) for ext in image_extensions):
        return False
    
    # Skip document files
    doc_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.zip', '.rar', '.css']
    if any(url_lower.endswith(ext) for ext in doc_extensions):
        return False
    
    # Skip media files
    media_extensions = ['.mp4', '.mp3', '.avi', '.mov', '.wav', '.flv']
    if any(url_lower.endswith(ext) for ext in media_extensions):
        return False
    
    # Skip common non-content paths
    skip_paths = ['/feed','/api/', '/admin/', '/wp-admin/', '/assets/', '/static/', '/images/', '/img/', '/css/', '/js/', '/wp-content/', '/administrator/', 'wp-json']
    if any(path in url_lower for path in skip_paths):
        return False
     # Don't skip pagination paths
    pagination_indicators = ['/page/', '?page=', '&page=', '/p/', '?p=', '&p=']
    
    # If it's a pagination URL, allow it
    if any(indicator in url_lower for indicator in pagination_indicators):
        return True
    
    return True

def extract_links_from_html(html: str, base_url: str) -> Set[str]:
    """Extract all links from HTML content"""
    links = set()
    
    # Find all href attributes
    href_pattern = r'href=["\']([^"\']+)["\']'
    matches = re.findall(href_pattern, html, re.IGNORECASE)
    
    for match in matches:
        # Convert relative URLs to absolute
        absolute_url = urljoin(base_url, match)
        normalized_url = normalize_url(absolute_url)
        
        # Only include valid content URLs from same domain
        if is_same_domain(normalized_url, base_url) and is_valid_content_url(normalized_url):
            links.add(normalized_url)
    
    return links

async def crawl_single_page(session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    """Crawl a single page with rate limiting"""
    async with semaphore:
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with session.get(url, timeout=timeout, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    
                    # Extract title
                    title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
                    title = title_match.group(1).strip() if title_match else "No title"
                    
                    # Extract meta description
                    meta_desc_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']', html, re.IGNORECASE)
                    meta_description = meta_desc_match.group(1).strip() if meta_desc_match else ""
                    
                    # Extract main content (remove scripts, styles, etc.)
                    cleaned_content = extract_main_content(html, url)
                    
                    return {
                        "url": url,
                        "title": title,
                        "meta_description": meta_description,
                        "content": cleaned_content,
                        "html": html,
                        "success": True,
                        "timestamp": datetime.now().isoformat(),
                        "status_code": response.status
                    }
                else:
                    return {
                        "url": url,
                        "success": False,
                        "error": f"HTTP {response.status}",
                        "timestamp": datetime.now().isoformat(),
                        "status_code": response.status
                    }
                    
        except asyncio.TimeoutError:
            return {
                "url": url,
                "success": False,
                "error": "Timeout after 30 seconds",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "url": url,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

def extract_main_content(html: str, url: str) -> str:
    """Extract main content from HTML, removing scripts, styles, etc."""
    if not html:
        return ""
    
    # Remove script and style tags
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove HTML tags but keep text content
    text = re.sub(r'<[^>]+>', ' ', html)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Limit content length to prevent memory issues
    max_length = 50000  # 50K characters per page
    if len(text) > max_length:
        text = text[:max_length] + "... [Content truncated]"
    
    return text

async def crawl_links_and_content(start_url: str, max_depth: int = 1, max_pages: int = 50) -> List[Dict[str, Any]]:
    """Enhanced crawl function with better performance and deduplication"""
    try:
        start_time = time.time()
        
        # Validate start URL
        if not start_url.startswith(('http://', 'https://')):
            return [{
                "success": False,
                "error": "Invalid URL format. Must start with http:// or https://",
                "url": start_url,
                "timestamp": datetime.now().isoformat()
            }]
        
        # Initialize tracking variables
        crawled_urls: Set[str] = set()
        to_crawl: List[str] = [normalize_url(start_url)]
        all_results: List[Dict[str, Any]] = []
        
        # Create semaphore for rate limiting (max 10 concurrent requests)
        max_concurrent = min(10, max_pages)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Configure session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=max_concurrent,
            limit_per_host=5,
            keepalive_timeout=40,
            enable_cleanup_closed=True
        )
        
        async with aiohttp.ClientSession(connector=connector) as session:
            
            for depth in range(max_depth):
                if not to_crawl or len(all_results) >= max_pages:
                    break
                
                st.info(f"🕷️ Crawling depth {depth + 1}/{max_depth} - {len(to_crawl)} URLs to process")
                
                # Process URLs in batches to avoid memory issues
                remaining_slots = max_pages - len(all_results)
                if remaining_slots <= 0:
                    break
                    
                batch_size = min(100, remaining_slots, len(to_crawl))  # Increased batch size
                current_batch = to_crawl[:batch_size]
                to_crawl = to_crawl[batch_size:]
                
                # Filter out already crawled URLs
                current_batch = [url for url in current_batch if url not in crawled_urls]
                
                if not current_batch:
                    continue
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Crawl current batch
                tasks = [crawl_single_page(session, url, semaphore) for url in current_batch]
                
                # Process with progress updates
                completed_results = []
                for i, task in enumerate(asyncio.as_completed(tasks)):
                    result = await task
                    completed_results.append(result)
                    
                    # Update progress
                    progress = (i + 1) / len(tasks)
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {i + 1}/{len(tasks)} pages in batch")
                    
                    # Add small delay to be respectful to servers
                    await asyncio.sleep(0.1)
                
                # Process results and find new links
                next_urls = []
                for result in completed_results:
                    url = result['url']
                    crawled_urls.add(url)
                    all_results.append(result)
                    
                    # Extract links from successful pages for next depth level
                    if result['success'] and depth < max_depth - 1 and 'html' in result:
                        try:
                            page_links = extract_links_from_html(result['html'], url)
                            new_links_count = 0
                            for link in page_links:
                                if (link not in crawled_urls and 
                                    link not in [item for sublist in [to_crawl, next_urls] for item in sublist]):
                                    next_urls.append(link)
                                    new_links_count += 1
                                    
                            if new_links_count > 0:
                                st.success(f"   📎 Found {new_links_count} new links from {url}")
                                
                        except Exception as e:
                            st.warning(f"Error extracting links from {url}: {str(e)}")
                
                # Add new URLs to crawl queue for next depth
                if next_urls:
                    to_crawl.extend(next_urls)
                    st.info(f"✅ Added {len(next_urls)} new URLs for next depth level")
                else:
                    st.warning("No new URLs found for next depth level")
                
                # Clean up progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Memory management - remove HTML content from results to save memory
                for result in completed_results:
                    if 'html' in result:
                        del result['html']
        
        # Final statistics with detailed breakdown
        elapsed_time = time.time() - start_time
        successful = sum(1 for r in all_results if r.get('success', False))
        failed = len(all_results) - successful
        
        st.success(f"✅ Crawling completed in {elapsed_time:.2f} seconds")
        st.info(f"📊 Results: {successful} successful + {failed} failed = {len(all_results)} total pages")
        st.info(f"🔍 Unique URLs discovered: {len(crawled_urls)}")
        
        return all_results
        
    except Exception as e:
        return [{
            "success": False,
            "error": f"Crawling failed: {str(e)}",
            "url": start_url,
            "timestamp": datetime.now().isoformat()
        }]

def show_crawl_interface():
    st.markdown("### 🕷️ Enhanced Website Crawler")
    st.info("Automatically crawl a website with advanced filtering and performance optimization.")
    
    # Input form
    url = st.text_input("Website URL to crawl:", placeholder="https://example.com")
    
    # Advanced options
    with st.expander("🚀 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            max_depth = st.number_input("Max crawl depth", min_value=1, max_value=5, value=1, step=1)
            max_pages = st.number_input("Max pages to crawl", min_value=1, max_value=500, value=50, step=10)
        
        with col2:
            enable_debug = st.checkbox("🐛 Enable Debug Mode", value=False)
            st.info("**Features:**\n- Same domain only\n- Excludes images & documents\n- Auto deduplication\n- Rate limiting\n- Memory optimization")
    
    # Add debug info
    if enable_debug:
        st.warning("🐛 Debug mode enabled - will show detailed crawling process")
    
    # Crawl button
    crawl_button = st.button("🚀 Start Enhanced Crawler", type="primary")

    if crawl_button and url:
        if url.startswith(('http://', 'https://')):
            
            # Show warning for large crawls
            if max_pages > 100:
                st.warning(f"⚠️ You're about to crawl {max_pages} pages. This may take several minutes.")
                
            with st.spinner("🕷️ Enhanced crawling in progress..."):      
                # Run async crawl function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Pass debug flag to crawler
                if 'enable_debug' in locals() and enable_debug:
                    st.session_state.debug_mode = True
                else:
                    st.session_state.debug_mode = False
                    
                results = loop.run_until_complete(crawl_links_and_content(url, max_depth, max_pages))
                loop.close()
                
                # Save results to session state
                st.session_state.crawl_results = results
                st.session_state.last_crawled_url = url

                # Show results
                if results and any(r.get('success', False) for r in results):
                    st.success(f"✔️ Enhanced crawling completed!")
                    st.balloons()
                    
                    # Save to database if user exists
                    if hasattr(st.session_state, 'user') and st.session_state.user:
                        for result in results:
                            try:
                                save_crawl_result(st.session_state.user.id, result)
                            except:
                                pass  # Ignore database errors

                    # Create combined content
                    combined_content = ""
                    successful_results = [r for r in results if r.get('success', False)]
                    
                    for result in successful_results:
                        combined_content += f"=== URL: {result['url']} ===\n"
                        combined_content += f"Title: {result.get('title', 'No title')}\n"
                        combined_content += f"Meta Description: {result.get('meta_description', 'No description')}\n"
                        combined_content += f"Content:\n{result.get('content', 'No content')}\n\n"
                    
                    # Show preview
                    with st.expander("👀 Content Preview"):
                        preview_length = 3000
                        preview_text = combined_content[:preview_length]
                        if len(combined_content) > preview_length:
                            preview_text += f"\n\n... [Showing first {preview_length} characters of {len(combined_content)} total]"
                        st.text_area("Content Preview", preview_text, height=300)
                    
                    # Show detailed summary
                    st.subheader("📊 Crawl Results")
                    
                    # Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    successful = len(successful_results)
                    failed = len(results) - successful
                    
                    col1.metric("Total Pages", len(results))
                    col2.metric("✅ Successful", successful)
                    col3.metric("❌ Failed", failed)
                    col4.metric("📄 Content Size", f"{len(combined_content):,} chars")
                    
                    # Show URLs with status
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
                    
                    # Download options
                    st.subheader("⬇️ Download Options")
                    
                    col_dl1, col_dl2 = st.columns(2)
                    
                    with col_dl1:
                        # Download combined content
                        if combined_content:
                            st.download_button(
                                label="📄 Download All Content (TXT)",
                                data=combined_content,
                                file_name=f"crawl_content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                key="download_combined_content"
                            )
                    
                    with col_dl2:
                        # Download URL list
                        url_list = "\n".join([f"{r['url']} - {'Success' if r.get('success') else 'Failed'}" for r in results])
                        st.download_button(
                            label="🔗 Download URL List",
                            data=url_list,
                            file_name=f"crawl_urls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            key="download_url_list"
                        )
                        
                else:
                    st.error("❌ Enhanced crawling failed. Please check the URL and try again.")
                    if results and results[0].get('error'):
                        st.error(f"Error details: {results[0]['error']}")
        else:
            st.warning("⚠️ Please enter a valid URL starting with http:// or https://")

def show_crawl_history():
    
    st.info("View, filter, and download records of previously crawled websites and content.")
    
    if st.session_state.user:
        history = get_user_crawl_history(st.session_state.user.id)
        
        if history:
            # Clear all history button
            if st.button("🗑️ Clear All History", type="secondary"):
                if clear_all_history(st.session_state.user.id):
                    st.success("All history cleared!")
                    st.rerun()
                else:
                    st.error("Failed to clear history")
            
            for item in history:
                with st.expander(f"{'✔️' if item['success'] else '❌'} {item['url']} - {item['created_at'][:19]}"):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**Title:** {item['title']}")
                        st.write(f"**URL:** {item['url']}")
                        st.write(f"**Timestamp:** {item['created_at']}")
                        if not item['success']:
                            st.error(f"Error: {item['error_message']}")
                    
                    with col2:
                        if item['success']:
                            st.metric("Content Length", f"{len(item['content'])} chars")
                    with col3:
                        # Re-crawl button
                        if st.button("🔄 Re-crawl", key=f"recrawl_{item['id']}"):
                            with st.spinner(f"Re-crawling {item['url']}..."):
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                new_result = loop.run_until_complete(crawl_url(item['url']))
                                loop.close()
                                
                                if new_result["success"]:
                                    save_crawl_result(st.session_state.user.id, new_result)
                                    st.success("Re-crawled successfully!")
                                    st.rerun()
                        
                        # Delete button
                        if st.button("❌ Delete", key=f"delete_{item['id']}"):
                            if delete_crawl_result(item['id']):
                                st.success("Item deleted!")
                                st.rerun()
                            else:
                                st.error("Failed to delete item")
                        
                        # View content button
                        if st.button("📄 View Content", key=f"view_{item['id']}"):
                            st.text_area("Content:", item['content'], height=200)
        else:
            st.info("No crawl history found. Start crawling some websites!")
    else:
        st.error("User not found")

if __name__ == "__main__":
    main()