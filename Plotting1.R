##########################
library(ggplot2)
library(readxl)
library(openxlsx)
library(gridExtra)
library(dplyr)

############Auxiliary functions##############
bar_plots <- function(data,summary_prob){
  
  #total_data$Annotation <- ifelse(total_data$overall_accuracies_mean > 60, "*","")
  
  plt <- ggplot( summary_prob,aes(x = as.factor(hormones_combination), y = Mean_prob, fill = true_status)) +
    geom_col(position = "dodge", width = 0.5, alpha = 0.7, color = "black", size = 0.1)+
    geom_errorbar(aes(ymin = Mean_prob - SE_prob, ymax = Mean_prob + SE_prob),
                  position =  position_dodge(width = 0.5), width = 0.2) #+ geom_text(aes(label = Annotation), 
                                                                                   # position = position_dodge(width = 0.8), 
                                                                                    #vjust = -0.5, size = 7, color = "red") +
    #scale_fill_manual(values = c("A" = "purple","B" = "purple", "P" = "purple", "I" = "purple","all-A" = "magenta","all-B" = "magenta", "all-P" = "magenta", "all-I" = "magenta" ))
  
  plt <- plt + theme_classic() + theme(legend.position.inside = c(0.8,0.85),plot.title = element_text(size = 15),
                                       plot.subtitle=element_text(size=15),
                                       axis.text = element_text(size = 8),
                                       
                                       legend.text=element_text(size=10),legend.title = element_blank(),axis.title = element_text(size = 14),
                                       legend.key.size = unit(1, "lines")) +
    geom_hline(yintercept = 50, color = "blue", linetype = "dashed", size = 1) + 
    labs(title = paste('alpha vs dominant',data[1,]$sex,sep=" "))+
    #labs(title = paste("Prediction for", total_data[1,]$sex, hormones, sep=" "), subtitle =  "* means overall accuracy > 60%",) + ylab("accuracy %")+
    xlab(" ")+ ylab("probability(%) to find true alpha or true submissive") +
    scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, by = 10))
  
  
  windows()
  grid.arrange(plt,nrow=1,ncol=1)
  
  return(plt)
}



bar_plots_acc <- function(data, summary_prob_ac){
  
  #total_data$Annotation <- ifelse(total_data$overall_accuracies_mean > 60, "*","")
  
  plt <- ggplot(summary_prob_ac,aes(x = as.factor(Hormones), y = Mean_prob)) +
    geom_col(position = "dodge", width = 0.2, alpha = 0.7, color = "black", size = 0.1)+
    geom_errorbar(aes(ymin = Mean_prob - SE_prob, ymax = Mean_prob + SE_prob),
                  position =  position_dodge(width = 0.5), width = 0.1) #+ geom_text(aes(label = Annotation), 
  # position = position_dodge(width = 0.8), 
  #vjust = -0.5, size = 7, color = "red") +
  #scale_fill_manual(values = c("A" = "purple","B" = "purple", "P" = "purple", "I" = "purple","all-A" = "magenta","all-B" = "magenta", "all-P" = "magenta", "all-I" = "magenta" ))
  
  plt <- plt + theme_classic() + theme(legend.position.inside = c(0.8,0.85),plot.title = element_text(size = 15),
                                       plot.subtitle=element_text(size=15),
                                       axis.text = element_text(size = 8),
                                       
                                       legend.text=element_text(size=10),legend.title = element_blank(),axis.title = element_text(size = 14),
                                       legend.key.size = unit(1, "lines")) +
    geom_hline(yintercept = 50, color = "blue", linetype = "dashed", size = 1) + 
    labs(title = paste('alpha vs dominant',data[1,]$sex,sep=" "))+
    #labs(title = paste("Prediction for", total_data[1,]$sex, hormones, sep=" "), subtitle =  "* means overall accuracy > 60%",) + ylab("accuracy %")+
    xlab(" ")+ ylab("accuracy(%) model") +
    scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, by = 10))
  
  
  windows()
  grid.arrange(plt,nrow=1,ncol=1)
  
  return(plt)
}

#########################
###Users data
input_file <- 'F:/Ruti/AnalysisWithPython/data_to_use.xlsx'
OutputFile <- 'F:/Ruti/AnalysisWithPython/Graphs/'
# sheet_name <- 'proba_alpha_vs_sub_male'
# sheet_name_ac <- 'accuracy_alpha_vs_sub_male'
# title <- 'alpha_vs_sub_male.pdf'

sheet_name <- 'proba_alpha_vs_sub_female'
sheet_name_ac <- 'accuracy_alpha_vs_sub_female'
title <- 'alpha_vs_sub_female.pdf'

plt<-list()
  
#Read data from Excel
data<- read_excel(input_file , sheet = sheet_name)
data_ac<- read_excel(input_file , sheet = sheet_name_ac)

######## plot bar plot  #########
#add a new column with probability to plot
data$probability <- 1
data[data$true_status == "submissive",]$probability <- data[data$true_status == "submissive",]$submissive
data[data$true_status == "alpha",]$probability <- data[data$true_status == "alpha",]$alpha

### summary the data

summary_prob <- data %>% 
  group_by(hormones_combination,true_status) %>%
  summarise( Mean_prob = mean(probability*100), SE_prob = sd(probability*100)/sqrt(n()))

summary_prob_ac <- data_ac %>% 
  group_by(Hormones) %>%
  summarise( Mean_prob = mean(accuracy*100), SE_prob = sd(accuracy*100)/sqrt(n()))



plt1 <- bar_plots(data,summary_prob)

plt2 <- bar_plots_acc(data_ac, summary_prob_ac)

combined_plot <- grid.arrange(plt1, plt2, nrow = 2)

# Save the combined plot as a PDF
pdf(paste(OutputFile,title,sep=""), width = 15, height = 12)  # Adjust dimensions as needed
grid.arrange(plt1, plt2, nrow = 2)  # Recreate the combined plot inside the PDF
dev.off()
