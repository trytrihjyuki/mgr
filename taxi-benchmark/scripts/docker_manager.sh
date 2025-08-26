#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Function to show main menu
show_menu() {
    clear
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║          ${GREEN}🚕 Taxi Benchmark Docker Manager ${CYAN}                  ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Available Operations:${NC}"
    echo ""
    echo -e "  ${BLUE}[1]${NC} 📦 ${GREEN}Build & Push${NC} - Build Docker image and push to ECR"
    echo -e "  ${BLUE}[2]${NC} ⬇️  ${GREEN}Pull${NC} - Pull Docker image from ECR"
    echo -e "  ${BLUE}[3]${NC} 🚀 ${GREEN}Run Experiment${NC} - Run experiment in Docker"
    echo -e "  ${BLUE}[4]${NC} 🏃 ${GREEN}Quick Local Run${NC} - Build and run locally (dev mode)"
    echo -e "  ${BLUE}[5]${NC} 📊 ${GREEN}Example Experiments${NC} - Show example commands"
    echo -e "  ${BLUE}[6]${NC} 🔍 ${GREEN}Check Environment${NC} - Verify Docker and AWS setup"
    echo -e "  ${BLUE}[7]${NC} 🧹 ${GREEN}Clean Up${NC} - Remove local Docker images and containers"
    echo -e "  ${BLUE}[8]${NC} ❌ ${GREEN}Exit${NC}"
    echo ""
    echo -n -e "${MAGENTA}Select an option [1-8]: ${NC}"
}

# Function to check environment
check_environment() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  Environment Check${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    # Check Docker
    echo -e "${YELLOW}Docker:${NC}"
    if command -v docker &> /dev/null; then
        echo -e "  ${GREEN}✓ Docker installed${NC}"
        docker_version=$(docker --version | cut -d' ' -f3 | sed 's/,$//')
        echo -e "  ${BLUE}  Version: ${docker_version}${NC}"
    else
        echo -e "  ${RED}✗ Docker not installed${NC}"
    fi
    
    # Check AWS CLI
    echo ""
    echo -e "${YELLOW}AWS CLI:${NC}"
    if command -v aws &> /dev/null; then
        echo -e "  ${GREEN}✓ AWS CLI installed${NC}"
        aws_version=$(aws --version | cut -d' ' -f1 | cut -d'/' -f2)
        echo -e "  ${BLUE}  Version: ${aws_version}${NC}"
    else
        echo -e "  ${RED}✗ AWS CLI not installed${NC}"
    fi
    
    # Check AWS credentials
    echo ""
    echo -e "${YELLOW}AWS Configuration:${NC}"
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "")
    if [ -n "$AWS_ACCOUNT_ID" ]; then
        echo -e "  ${GREEN}✓ AWS credentials configured${NC}"
        echo -e "  ${BLUE}  Account ID: ${AWS_ACCOUNT_ID}${NC}"
        echo -e "  ${BLUE}  Region: ${AWS_REGION:-eu-north-1}${NC}"
    else
        echo -e "  ${RED}✗ AWS credentials not configured${NC}"
    fi
    
    # Check .env file
    echo ""
    echo -e "${YELLOW}Environment File:${NC}"
    if [ -f .env ]; then
        echo -e "  ${GREEN}✓ .env file found${NC}"
        if grep -q "S3_BUCKET=" .env; then
            S3_BUCKET=$(grep "^S3_BUCKET=" .env | cut -d'=' -f2 | tr -d '"')
            echo -e "  ${BLUE}  S3 Bucket: ${S3_BUCKET}${NC}"
        fi
    else
        echo -e "  ${YELLOW}⚠ .env file not found (using defaults)${NC}"
    fi
    
    echo ""
}

# Function to show example experiments
show_examples() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  Example Experiments${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    echo -e "${YELLOW}1. Full experiment with all methods (Manhattan & Brooklyn):${NC}"
    echo -e "${GREEN}./scripts/run_docker.sh -- \\
    --processing-date 2019-10-06 \\
    --vehicle-type green \\
    --boroughs Manhattan Brooklyn \\
    --methods LP MinMaxCostFlow MAPS LinUCB \\
    --num-iter 100 \\
    --start-hour 10 \\
    --end-hour 12${NC}"
    
    echo ""
    echo -e "${YELLOW}2. Quick test with single method (Queens):${NC}"
    echo -e "${GREEN}./scripts/run_docker.sh -- \\
    --processing-date 2019-10-06 \\
    --vehicle-type yellow \\
    --boroughs Queens \\
    --methods MinMaxCostFlow \\
    --num-iter 10 \\
    --start-hour 14 \\
    --end-hour 15${NC}"
    
    echo ""
    echo -e "${YELLOW}3. Baseline comparison (MAPS vs LinUCB):${NC}"
    echo -e "${GREEN}./scripts/run_docker.sh -- \\
    --processing-date 2019-10-06 \\
    --vehicle-type green \\
    --boroughs Manhattan \\
    --methods MAPS LinUCB \\
    --num-iter 50 \\
    --time-delta 30${NC}"
    
    echo ""
    echo -e "${YELLOW}4. Full day experiment (all boroughs):${NC}"
    echo -e "${GREEN}./scripts/run_docker.sh -- \\
    --processing-date 2019-10-06 \\
    --vehicle-type yellow \\
    --boroughs Manhattan Brooklyn Queens Bronx \\
    --methods MinMaxCostFlow MAPS \\
    --num-iter 100 \\
    --start-hour 6 \\
    --end-hour 22${NC}"
    
    echo ""
}

# Function to clean up Docker resources
cleanup_docker() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  Docker Cleanup${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    # Remove containers
    echo -e "${YELLOW}Removing taxi-benchmark containers...${NC}"
    docker ps -a | grep taxi-benchmark | awk '{print $1}' | xargs -r docker rm -f 2>/dev/null || true
    echo -e "${GREEN}✓ Containers removed${NC}"
    
    # Remove images
    echo -e "${YELLOW}Removing taxi-benchmark images...${NC}"
    docker images | grep taxi-benchmark | awk '{print $3}' | xargs -r docker rmi -f 2>/dev/null || true
    echo -e "${GREEN}✓ Images removed${NC}"
    
    # Prune system
    echo -e "${YELLOW}Pruning Docker system...${NC}"
    docker system prune -f
    echo -e "${GREEN}✓ System pruned${NC}"
    
    echo ""
    echo -e "${GREEN}Cleanup completed!${NC}"
}

# Function to run quick local experiment
quick_local_run() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  Quick Local Run${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    echo -e "${YELLOW}This will build and run a quick test experiment locally.${NC}"
    echo ""
    read -p "Enter processing date (YYYY-MM-DD) [2019-10-06]: " date_input
    date_input=${date_input:-2019-10-06}
    
    read -p "Enter vehicle type (green/yellow) [green]: " vehicle_input
    vehicle_input=${vehicle_input:-green}
    
    read -p "Enter borough (Manhattan/Brooklyn/Queens/Bronx) [Manhattan]: " borough_input
    borough_input=${borough_input:-Manhattan}
    
    read -p "Enter method (LP/MinMaxCostFlow/MAPS/LinUCB) [MinMaxCostFlow]: " method_input
    method_input=${method_input:-MinMaxCostFlow}
    
    read -p "Enter number of iterations [10]: " iter_input
    iter_input=${iter_input:-10}
    
    echo ""
    echo -e "${YELLOW}Running experiment...${NC}"
    ${SCRIPT_DIR}/run_docker.sh --local --no-cache -- \
        --processing-date ${date_input} \
        --vehicle-type ${vehicle_input} \
        --boroughs ${borough_input} \
        --methods ${method_input} \
        --num-iter ${iter_input} \
        --start-hour 12 \
        --end-hour 13
}

# Main loop
while true; do
    show_menu
    read choice
    
    case $choice in
        1)
            echo ""
            read -p "Enter image tag [latest]: " tag
            tag=${tag:-latest}
            ${SCRIPT_DIR}/push_docker.sh ${tag}
            echo ""
            read -p "Press Enter to continue..."
            ;;
        2)
            echo ""
            read -p "Enter image tag to pull [latest]: " tag
            tag=${tag:-latest}
            ${SCRIPT_DIR}/pull_docker.sh ${tag}
            echo ""
            read -p "Press Enter to continue..."
            ;;
        3)
            echo ""
            echo -e "${YELLOW}Enter experiment arguments after the script name.${NC}"
            echo -e "${YELLOW}Example: ${GREEN}./scripts/run_docker.sh -- --processing-date 2019-10-06 --vehicle-type green --boroughs Manhattan --methods MinMaxCostFlow${NC}"
            echo ""
            echo -e "${YELLOW}Or use: ${GREEN}./scripts/run_docker.sh --help${NC} for full options"
            echo ""
            read -p "Press Enter to continue..."
            ;;
        4)
            quick_local_run
            echo ""
            read -p "Press Enter to continue..."
            ;;
        5)
            show_examples
            echo ""
            read -p "Press Enter to continue..."
            ;;
        6)
            check_environment
            echo ""
            read -p "Press Enter to continue..."
            ;;
        7)
            cleanup_docker
            echo ""
            read -p "Press Enter to continue..."
            ;;
        8)
            echo ""
            echo -e "${GREEN}Goodbye! 🚕${NC}"
            exit 0
            ;;
        *)
            echo ""
            echo -e "${RED}Invalid option. Please select 1-8.${NC}"
            sleep 2
            ;;
    esac
done 