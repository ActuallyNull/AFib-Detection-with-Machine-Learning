# ECG Classification AI - Deployment Guide

This guide covers deploying the React + FastAPI ECG classification application in various environments.

## 🚀 Quick Deployment

### 1. Docker Compose (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd actualibproject-deployment-reactfastapi

# Build and run
docker-compose up --build -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

**Access URLs:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### 2. Local Development

#### Frontend
```bash
cd frontend
npm install
npm run dev
```

#### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## ☁️ Cloud Deployment

### AWS Deployment

#### Option 1: AWS ECS with Fargate

1. **Create ECR repositories:**
```bash
aws ecr create-repository --repository-name ecg-frontend
aws ecr create-repository --repository-name ecg-backend
```

2. **Build and push images:**
```bash
# Frontend
docker build -t ecg-frontend ./frontend
docker tag ecg-frontend:latest $AWS_ACCOUNT.dkr.ecr.$REGION.amazonaws.com/ecg-frontend:latest
docker push $AWS_ACCOUNT.dkr.ecr.$REGION.amazonaws.com/ecg-frontend:latest

# Backend
docker build -t ecg-backend ./backend
docker tag ecg-backend:latest $AWS_ACCOUNT.dkr.ecr.$REGION.amazonaws.com/ecg-backend:latest
docker push $AWS_ACCOUNT.dkr.ecr.$REGION.amazonaws.com/ecg-backend:latest
```

3. **Create ECS cluster and services**
4. **Set up Application Load Balancer**
5. **Configure auto-scaling**

#### Option 2: AWS App Runner

```bash
# Deploy backend
aws apprunner create-service \
  --service-name ecg-backend \
  --source-configuration file://apprunner-backend.json

# Deploy frontend
aws apprunner create-service \
  --service-name ecg-frontend \
  --source-configuration file://apprunner-frontend.json
```

### Google Cloud Deployment

#### Option 1: Cloud Run

```bash
# Deploy backend
gcloud run deploy ecg-backend \
  --source ./backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

# Deploy frontend
gcloud run deploy ecg-frontend \
  --source ./frontend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Option 2: GKE (Kubernetes)

```bash
# Create cluster
gcloud container clusters create ecg-cluster \
  --num-nodes=3 \
  --zone=us-central1-a

# Deploy with kubectl
kubectl apply -f k8s/
```

### Azure Deployment

#### Option 1: Container Instances

```bash
# Deploy backend
az container create \
  --resource-group myResourceGroup \
  --name ecg-backend \
  --image ecg-backend:latest \
  --ports 8000

# Deploy frontend
az container create \
  --resource-group myResourceGroup \
  --name ecg-frontend \
  --image ecg-frontend:latest \
  --ports 80
```

#### Option 2: AKS (Kubernetes)

```bash
# Create cluster
az aks create \
  --resource-group myResourceGroup \
  --name ecg-cluster \
  --node-count 3

# Deploy with kubectl
kubectl apply -f k8s/
```

## 🐳 Kubernetes Deployment

### Create Kubernetes Manifests

#### `k8s/namespace.yaml`
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ecg-app
```

#### `k8s/backend-deployment.yaml`
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ecg-backend
  namespace: ecg-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ecg-backend
  template:
    metadata:
      labels:
        app: ecg-backend
    spec:
      containers:
      - name: backend
        image: ecg-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: PYTHONPATH
          value: "/app"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### `k8s/frontend-deployment.yaml`
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ecg-frontend
  namespace: ecg-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ecg-frontend
  template:
    metadata:
      labels:
        app: ecg-frontend
    spec:
      containers:
      - name: frontend
        image: ecg-frontend:latest
        ports:
        - containerPort: 80
        env:
        - name: VITE_API_URL
          value: "http://ecg-backend-service:8000"
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "200m"
```

#### `k8s/services.yaml`
```yaml
apiVersion: v1
kind: Service
metadata:
  name: ecg-backend-service
  namespace: ecg-app
spec:
  selector:
    app: ecg-backend
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
  type: ClusterIP
---
apiVersion: v1
kind: Service
metadata:
  name: ecg-frontend-service
  namespace: ecg-app
spec:
  selector:
    app: ecg-frontend
  ports:
  - protocol: TCP
    port: 80
    targetPort: 80
  type: LoadBalancer
```

#### `k8s/ingress.yaml`
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ecg-ingress
  namespace: ecg-app
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: ecg.yourdomain.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: ecg-backend-service
            port:
              number: 8000
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ecg-frontend-service
            port:
              number: 80
```

### Deploy to Kubernetes

```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n ecg-app
kubectl get services -n ecg-app
kubectl get ingress -n ecg-app
```

## 🔒 Production Security

### SSL/TLS Configuration

#### Using Let's Encrypt with Cert-Manager

```yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: ecg-cert
  namespace: ecg-app
spec:
  secretName: ecg-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
  - ecg.yourdomain.com
```

### Environment Variables

#### Production Environment
```bash
# Frontend
VITE_API_URL=https://api.yourdomain.com
VITE_ENABLE_LOGGING=false

# Backend
DEBUG=false
LOG_LEVEL=WARNING
CORS_ORIGINS=https://yourdomain.com
```

### Security Headers

Update nginx configuration in `frontend/nginx.conf`:
```nginx
# Security headers
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header X-Content-Type-Options "nosniff" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';" always;
```

## 📊 Monitoring & Logging

### Prometheus & Grafana

#### `k8s/monitoring.yaml`
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: ecg-app
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'ecg-backend'
      static_configs:
      - targets: ['ecg-backend-service:8000']
    - job_name: 'ecg-frontend'
      static_configs:
      - targets: ['ecg-frontend-service:80']
```

### Logging with ELK Stack

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: elasticsearch
  namespace: ecg-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: elasticsearch
  template:
    metadata:
      labels:
        app: elasticsearch
    spec:
      containers:
      - name: elasticsearch
        image: docker.elastic.co/elasticsearch/elasticsearch:7.17.0
        env:
        - name: discovery.type
          value: single-node
        ports:
        - containerPort: 9200
```

## 🔄 CI/CD Pipeline

### GitHub Actions

#### `.github/workflows/deploy.yml`
```yaml
name: Deploy ECG App

on:
  push:
    branches: [ main ]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v1
    
    - name: Build and push Docker images
      uses: docker/build-push-action@v2
      with:
        context: ./frontend
        push: true
        tags: ${{ secrets.ECR_REGISTRY }}/ecg-frontend:latest
        cache-from: type=gha
        cache-to: type=gha,mode=max
    
    - name: Deploy to Kubernetes
      uses: steebchen/kubectl@v2
      with:
        config: ${{ secrets.KUBE_CONFIG_DATA }}
        command: apply -f k8s/
```

### GitLab CI

#### `.gitlab-ci.yml`
```yaml
stages:
  - build
  - deploy

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t ecg-frontend ./frontend
    - docker build -t ecg-backend ./backend
    - docker push $CI_REGISTRY_IMAGE/ecg-frontend:$CI_COMMIT_SHA
    - docker push $CI_REGISTRY_IMAGE/ecg-backend:$CI_COMMIT_SHA

deploy:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl set image deployment/ecg-frontend frontend=$CI_REGISTRY_IMAGE/ecg-frontend:$CI_COMMIT_SHA
    - kubectl set image deployment/ecg-backend backend=$CI_REGISTRY_IMAGE/ecg-backend:$CI_COMMIT_SHA
    - kubectl rollout status deployment/ecg-frontend
    - kubectl rollout status deployment/ecg-backend
```

## 🧪 Testing Deployment

### Health Checks

```bash
# Test backend health
curl https://api.yourdomain.com/health

# Test frontend
curl https://yourdomain.com

# Test model info
curl https://api.yourdomain.com/model-info
```

### Load Testing

```bash
# Install k6
curl -L https://github.com/grafana/k6/releases/download/v0.40.0/k6-v0.40.0-linux-amd64.tar.gz | tar xz

# Run load test
k6 run load-test.js
```

#### `load-test.js`
```javascript
import http from 'k6/http';
import { check } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 10 },
    { duration: '5m', target: 10 },
    { duration: '2m', target: 0 },
  ],
};

export default function() {
  let res = http.get('https://api.yourdomain.com/health');
  check(res, { 'status is 200': (r) => r.status === 200 });
}
```

## 🔧 Troubleshooting

### Common Issues

1. **Model Loading Failures**
   ```bash
   # Check model file exists
   ls -la model.keras
   
   # Check model path in backend
   kubectl logs deployment/ecg-backend -n ecg-app
   ```

2. **CORS Issues**
   ```bash
   # Check CORS configuration
   curl -H "Origin: https://yourdomain.com" \
        -H "Access-Control-Request-Method: POST" \
        -H "Access-Control-Request-Headers: Content-Type" \
        -X OPTIONS https://api.yourdomain.com/predict
   ```

3. **Memory Issues**
   ```bash
   # Check resource usage
   kubectl top pods -n ecg-app
   
   # Scale up if needed
   kubectl scale deployment ecg-backend --replicas=5 -n ecg-app
   ```

### Performance Optimization

1. **Enable Caching**
   ```nginx
   # Add to nginx.conf
   location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
       expires 1y;
       add_header Cache-Control "public, immutable";
   }
   ```

2. **Database Integration**
   ```yaml
   # Add PostgreSQL for storing predictions
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: postgres
     namespace: ecg-app
   spec:
     replicas: 1
     selector:
       matchLabels:
         app: postgres
     template:
       metadata:
         labels:
           app: postgres
       spec:
         containers:
         - name: postgres
           image: postgres:13
           env:
           - name: POSTGRES_DB
             value: ecg_db
           - name: POSTGRES_USER
             value: ecg_user
           - name: POSTGRES_PASSWORD
             valueFrom:
               secretKeyRef:
                 name: postgres-secret
                 key: password
   ```

## 📈 Scaling

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ecg-backend-hpa
  namespace: ecg-app
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ecg-backend
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

This deployment guide provides comprehensive instructions for deploying the ECG classification application in various environments, from local development to production cloud deployments.
