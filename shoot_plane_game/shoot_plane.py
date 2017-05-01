# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:05:00 2013

@author: Leo
"""

import pygame
from sys import exit
# 导入一些常用的变量和函数
from pygame.locals import *
from gameEntity import *
import random

class ShootPlaneGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('飞机大战')
        filename = 'shoot_plane_game/resources/image/shoot.png'
        plane_img = pygame.image.load(filename)

        # 设置玩家相关参数
        player_rect = []
        player_rect.append(pygame.Rect(0, 99, 102, 126))        # 玩家精灵图片区域
        player_rect.append(pygame.Rect(165, 360, 102, 126))

        player_rect.append(pygame.Rect(165, 234, 102, 126))     # 玩家爆炸精灵图片区域
        player_rect.append(pygame.Rect(330, 624, 102, 126))
        player_rect.append(pygame.Rect(330, 498, 102, 126))
        player_rect.append(pygame.Rect(432, 624, 102, 126))
        player_pos = [(SCREEN_WIDTH - PLAYER_WIDTH) /2, SCREEN_HEIGHT - PLAYER_HEIGHT]      # 玩家初始位置
        self.player = Player(plane_img, player_rect, player_pos)

        # 定义子弹对象使用的surface相关参数
        bullet_rect = pygame.Rect(1004, 987, 9, 21)
        self.bullet_img = plane_img.subsurface(bullet_rect)

        # 定义小型敌机对象使用的surface相关参数
        self.enemy_small_rect = pygame.Rect(534, 612, 57, 43)
        self.enemy_small_img = plane_img.subsurface(self.enemy_small_rect)
        # 小型敌机被击中精灵图片
        self.enemy_small_down_imgs = []
        self.enemy_small_down_imgs.append(plane_img.subsurface(pygame.Rect(267, 347, 57, 43)))
        self.enemy_small_down_imgs.append(plane_img.subsurface(pygame.Rect(873, 697, 57, 43)))
        self.enemy_small_down_imgs.append(plane_img.subsurface(pygame.Rect(267, 296, 57, 43)))
        self.enemy_small_down_imgs.append(plane_img.subsurface(pygame.Rect(930, 697, 57, 43)))


        self.enemies = pygame.sprite.Group()

        # 存储被击毁的飞机，用来渲染击毁精灵动画
        self.enemies_down = pygame.sprite.Group()
        self.shoot_frequency = 0
        self.player_down_index = 16
        self.score = 0
        self.clock = pygame.time.Clock()
        self.fps = 30


    def frame_step(self, input_actions):   
        reward = 0.1
        terminal = False

        # 控制发射子弹频率,并发射子弹
        if not self.player.is_hit:
            if self.shoot_frequency % 10 == 0:
                self.player.shoot(self.bullet_img)
                self.shoot_frequency = 0
            self.shoot_frequency += 1

        # 生成敌机
        rint = random.randint(1, 500)
        if rint % 30 == 0:
            enemy_small_pos = [random.randint(0, SCREEN_WIDTH - self.enemy_small_rect.width), 0]
            enemy_small = Enemy_Small(self.enemy_small_img, self.enemy_small_down_imgs, enemy_small_pos)
            self.enemies.add(enemy_small)


        # 移动子弹，若超出窗口范围则删除
        for bullet in self.player.bullets:
            bullet.move()
            if bullet.rect.bottom < 0:
                self.player.bullets.remove(bullet)

        # 移动敌机，若超出窗口范围则删除
        for enemy in self.enemies:
            enemy.move()
            # 判断玩家是否被击中
            if pygame.sprite.collide_circle(enemy, self.player):
                self.enemies_down.add(enemy)
                self.enemies.remove(enemy)
                self.player.is_hit = True
                break
            if enemy.rect.top < 0:
                self.enemies.remove(enemy)

        # 将被击中的敌机对象添加到击毁敌机Group中，用来渲染击毁动画
        enemies_shooted = pygame.sprite.groupcollide(self.enemies, self.player.bullets, 0, 1)
        if len(enemies_shooted) > 0:
        	reward = 1
        for enemy_down in enemies_shooted:
            enemy_down.shootcount = enemy_down.shootcount - 1
            if enemy_down.isDown():
                self.enemies_down.add(enemy_down)
                self.enemies.remove(enemy_down)

        # 绘制背景
        self.screen.fill(0)

        # 绘制玩家飞机
        if not self.player.is_hit:
            self.screen.blit(self.player.image[self.player.img_index], self.player.rect)
            # 更换图片索引使飞机有动画效果
            self.player.img_index = self.shoot_frequency / 8
        else:
            self.player.img_index = self.player_down_index / 8
            self.screen.blit(self.player.image[self.player.img_index], self.player.rect)
            self.player_down_index += 1
            if self.player_down_index > 47:
	            terminal = True
	            reward = -1
	            self.__init__()


        # 绘制子弹和敌机
        self.player.bullets.draw(self.screen)
        self.enemies.draw(self.screen)


        # 若玩家被击中，则无效
        # if not self.player.is_hit:
        #     if key_pressed[K_w] or key_pressed[K_UP]:
        #         self.player.moveUp()
        #     if key_pressed[K_s] or key_pressed[K_DOWN]:
        #         self.player.moveDown()
        #     if key_pressed[K_a] or key_pressed[K_LEFT]:
        #         self.player.moveLeft()
        #     if key_pressed[K_d] or key_pressed[K_RIGHT]:
        #         self.player.moveRight()
        if not self.player.is_hit:
            if input_actions[0]:
                self.player.moveLeft()
            if input_actions[1]:
                self.player.moveRight()
            if input_actions[2]:
                self.player.moveUp()
            if input_actions[3]:
                self.player.moveDown()
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        self.clock.tick(self.fps)
        #print self.upperPipes[0]['y'] + PIPE_HEIGHT - int(BASEY * 0.2)
        return image_data, reward, terminal
