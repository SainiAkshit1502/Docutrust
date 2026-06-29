FROM nginx:alpine

# Remove default nginx configurations and static assets
RUN rm -rf /usr/share/nginx/html/* /etc/nginx/conf.d/default.conf

# Copy custom reverse proxy config into Nginx configuration directory
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy static frontend assets into nginx HTML directory
COPY . /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
