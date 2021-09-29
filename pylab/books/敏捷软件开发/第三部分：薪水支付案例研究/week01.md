## 敏捷软件开发第三部分第一周读书笔记

目标：第13,14,15,16,17章 ~ 35页

- Command模式
- Active Object模式
- 模板方法
- 策略模式
- Facade模式
- Mediator模式
- 单例模式
- Mono State模式

###  COMMAND模式和ACTIVE OBJECT模式 ###

#### Command模式 ####

命令模式将”请求“封装成对象，以便使用不同的请求、队列或者日志来参数化其他对象，命令模式也支持可撤销的操作

#### Active Object模式 ####

Active Object 模式是Command模式的一种，是实现多线程控制的一项古老技术

### TEMPLATE METHOD模式和STRATEGY模式：继承与委托 ###

> 本章两个模式归纳了继承和委托之间的区别
> TEMPLATE METHOD模式使用继承来解决问题，而STRATEGY模式使用的则是委托

继承的意义：使用继承可以基于差异编程（program by difference）

继承的缺点：容易被过度使用，而且过度使用的代价非常高，所以建议：优先使用对象组合而不是继承，所以我们减少了对继承的使用，常常使用组合或者委托来代替它

TEMPLATE METHOD模式: 该模式把所有通用代码放入一个抽象基类的实现方法中

STRATEGY模式: STRATEGY模式使用了一种非常不同的方法来倒置通用算法和具体实现之间的依赖关系


### FACADE模式和MEDIATOR模式 ###

> 本章中论述的两个模式有着共同的目的。它们都把某种规约（policy）施加到另外一组对象上。FACADE模式从上面施加规约，而MEDIATOR模式则从下面施加规约

FACADE模式: 当想要为一组具有复杂且通用的接口的对象提供一个简单且特定的接口时，可以使用FACADE模式 

EDIATOR模式: MEDIATOR模式同样也施加规约。不过，FACADE模式是以可见且强制的方式来施加它的规约，而MEDIATOR模式则是以隐藏且自由的方式来施加它的规约

### SINGLETON模式和MONOSTATE模式 ###

SINGLETON模式: 通过Singleton模式，全局保证只有一个实例
 
MONOSTATE模式: MonoState并不限制创建对象的个数，但是它的状态却只有一个状态

#### NULL OBJECT模式 ####

为了获取某属性，但是有时候属性是None，那么需要你做异常处理， 而假如你想节省这样的条件过滤的代码，可以使用Null模式以减少对象是否为None的判断
