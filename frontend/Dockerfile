FROM node:18-alpine AS build

ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL

WORKDIR /app

# Copy package files and install dependencies.
# Using "npm ci" is recommended for reproducible builds.
COPY package.json package-lock.json ./
RUN npm ci

# Copy the rest of the application source code
COPY . .
# Build the application
RUN npm run build

# Stage 2: Production stage
FROM nginx:stable-alpine
COPY --from=build /app/dist /usr/share/nginx/html/
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]

