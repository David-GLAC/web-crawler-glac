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
from urllib.parse import urlparse
from collections import defaultdict

# MUST BE FIRST: Configure Streamlit page
st.set_page_config(
    page_title="Web Crawler App",
    page_icon="image/web-crawler.png",
    layout="wide"
)

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
            verbose=False
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
    col1, col2 = st.columns([3, 1])
    with col1:
        st.image("image/web-crawler.png")
        st.title("Web Crawler App")
    with col2:
        if st.button("Sign Out"):
            sign_out()
            st.session_state.authenticated = False
            st.session_state.user = None
            st.rerun()
    
    # User info
    if st.session_state.user:
        st.info(f"Welcome, {st.session_state.user.email} 👋")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌐 Crawl Website (URLs + Content)",
        "📄 Crawl Full Content (Single or Multiple URLs)",
        "🎯 Targeted Content Crawl (Title, Meta, Keywords...)",
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
    st.markdown("### Targeted Content Crawl")
    st.info("Extract specific elements such as `<title>`, meta description, keywords, headings, or custom CSS selectors.")

def show_crawl_url():
    st.markdown("### Crawl Full Content")
    st.info("Enter a single URL or upload a `.txt` file containing multiple URLs to extract the full page content.")

    # Input form
    with st.form("url_crawl_form"):
        url = st.text_input("Enter URL:", placeholder="https://example.com")
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
                    status_text.text(f"🕷️ Crawling {i+1}/{len(urls_to_crawl)}: {url}")
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
            st.balloons()
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
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                from reportlab.lib.styles import getSampleStyleSheet
                import io
                
                def generate_pdf():
                    buffer = io.BytesIO()
                    doc = SimpleDocTemplate(buffer, pagesize=letter)
                    styles = getSampleStyleSheet()
                    story = []
                    
                    for item in successful:
                        story.append(Paragraph(f"<b>URL:</b> {item['url']}", styles['Normal']))
                        story.append(Paragraph(f"<b>Title:</b> {item.get('title', 'No title')}", styles['Normal']))
                        story.append(Paragraph(f"<b>Meta Description:</b> {item.get('meta_description', 'No description')}", styles['Normal']))
                        story.append(Paragraph("<b>Content:</b>", styles['Normal']))
                        story.append(Paragraph(item.get('content', 'No content'), styles['Normal']))
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
                    
def show_crawl_interface():
    st.markdown("### Crawl Website (URLs + Content)")
    st.info("Automatically crawl a base URL and its internal links, collecting the full content of each page.")
    
    # Input form
    url = st.text_input("Website URL to crawl:", placeholder="https://example.com")
    
    # Advanced options
    with st.expander("🚀 Advanced Options"):
        max_depth = st.number_input("Max crawl depth", min_value=1, max_value=5, value=1, step=1)
        max_pages = st.number_input("Max number of pages to crawl", min_value=1, max_value=1000, value=50, step=20)
        
    crawl_button = st.button("Start crawler", type="primary")

    if crawl_button and url:
        if url.startswith(('http://', 'https://')):
            with st.spinner("🕷️ Crawling website and collecting content... This may take a while..."):      
                # Run async crawl function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(crawl_links_and_content(url, max_depth, max_pages))
                loop.close()
                
                # Save results to session state
                st.session_state.crawl_results = results
                st.session_state.last_crawled_url = url

                # Show results
                if results and any(r['success'] for r in results):
                    st.success(f"✔️ Crawling completed!")
                    st.balloons()
                    
                    # Save to database
                    if st.session_state.user:
                        for result in results:
                            save_crawl_result(st.session_state.user.id, result)

                    # Create combined content
                    combined_content = ""
                    for result in results:
                        if result['success']:
                            combined_content += f"=== URL: {result['url']} ===\n"
                            combined_content += f"Title: {result.get('title', 'No title')}\n"
                            combined_content += f"Meta Description: {result.get('meta_description', 'No description')}\n"
                            combined_content += f"Content:\n{result.get('content', 'No content')}\n\n"
                    
                    # Show preview
                    with st.expander("👀 Preview content..."):
                        st.text(combined_content[:2000] + "..." if len(combined_content) > 2000 else combined_content)
                    
                    # Show summary
                    st.subheader("📊 Summary")
                    col_left, col_right = st.columns(2)
                    
                    with col_left:
                        st.subheader("Crawled URLs")
                        for i, result in enumerate(results, 1):
                            status = "🔗" if result['success'] else "❌"
                            st.write(f"{i}. {status} {result['url']}")

                    with col_right:
                        st.subheader("Crawl Statistics")
                        successful = sum(1 for r in results if r['success'])
                        failed = len(results) - successful
                            
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Pages", len(results))
                        col2.metric("Successful", successful)
                        col3.metric("Failed", failed)
                        
                        # Download button
                        st.download_button(
                            label="⬇️ Download All Content as TXT",
                            data=combined_content,
                            file_name=f"combined_crawl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            key="download_combined_txt"
                        )
                else:
                    st.error("❌ Crawling failed. Please check the URL and try again.")
                    if results and 'error' in results[0]:
                        st.error(f"Error details: {results[0]['error']}")
        else:
            st.warning("Please enter a valid URL starting with http:// or https://")

def show_crawl_history():
    st.markdown("### Crawl History")
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